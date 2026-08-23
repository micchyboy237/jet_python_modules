# jet/adapters/llama_cpp/tasks/rag/eval/llm_as_a_judge/service.py
"""Production RAG service with real retrieval/generation via jet adapters."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from jet.adapters.llama_cpp.chunking_utils import truncate_texts
from jet.adapters.llama_cpp.hybrid_utils import hybrid_search
from jet.adapters.llama_cpp.llm_utils import achat
from jet.adapters.llama_cpp.model_utils import get_model_ctx_embd_size

from .evaluator import RAGEvaluator

logger = logging.getLogger(__name__)


class RAGService:
    """Fully implemented RAG service using jet adapters. No placeholders."""

    GENERATION_MODEL = "qwen3.5-uncensored:2b"
    MAX_CONTEXT_CHUNKS = 5

    def __init__(
        self,
        evaluator: RAGEvaluator,
        documents: list[str],
        generation_model: str | None = None,
    ):
        self.evaluator = evaluator
        self.documents = documents
        self.gen_model = generation_model or self.GENERATION_MODEL
        self._eval_queue: asyncio.Queue = asyncio.Queue()
        self._eval_worker_task: Optional[asyncio.Task] = None

        from jet.adapters.llama_cpp.embed_utils import embed

        logger.info(
            "📦 Pre-computing document embeddings for %d docs...", len(documents)
        )
        self._doc_embeddings = embed(documents, return_format="numpy")
        logger.info(
            "✅ Document embeddings ready (shape=%s)", self._doc_embeddings.shape
        )

    async def start(self):
        self._eval_worker_task = asyncio.create_task(self._eval_worker())
        logger.info("🚀 RAG evaluation background worker started")

    async def stop(self):
        if self._eval_worker_task:
            self._eval_worker_task.cancel()
            try:
                await self._eval_worker_task
            except asyncio.CancelledError:
                pass
            logger.info("🛑 Background eval worker stopped")

    async def query(self, user_query: str) -> dict:
        logger.info("🔎 Processing query: %r", user_query[:80])

        # Step 1: Retrieve
        contexts = await self._retrieve(user_query)
        logger.info("📚 Retrieved %d context chunks", len(contexts))

        # Step 2: Pre-generation gate
        gate = await self.evaluator.evaluate_pre_generation_gate(user_query, contexts)
        logger.info(
            "🚦 Pre-gen gate: precision=%.3f passed=%s tokens=%d",
            gate.contextual_precision or 0,
            gate.passed_gate,
            gate.total_eval_tokens,
        )

        if not gate.passed_gate:
            logger.warning(
                "⛔ Gate FAILED for query=%r — returning fallback", user_query[:60]
            )
            return {
                "answer": "I couldn't find sufficient information to answer reliably.",
                "confidence": "low",
                "debug": {"gate_failed": True, "precision": gate.contextual_precision},
            }

        # Step 3: Generate
        response = await self._generate(user_query, contexts)
        logger.info("💬 Generated response: %d chars", len(response))

        # Step 4: Queue async production eval
        await self._eval_queue.put((user_query, contexts, response))
        logger.debug(
            "📤 Queued production eval (queue_size=%d)", self._eval_queue.qsize()
        )

        return {"answer": response, "confidence": "high"}

    async def _retrieve(self, query: str) -> list[str]:
        """Real retrieval via hybrid_search with auto-fallback reranking."""
        logger.debug(
            "🔍 hybrid_search: query=%r, top_n=%d", query[:60], self.MAX_CONTEXT_CHUNKS
        )
        try:
            results = hybrid_search(
                query=query,
                documents=self.documents,
                top_n=self.MAX_CONTEXT_CHUNKS,
                doc_embeddings=self._doc_embeddings,
                normalize_scores=True,
            )
            logger.info(
                "✅ Retrieval complete: %d results, scores=[%s]",
                len(results),
                ", ".join(f"{r['score']:.3f}" for r in results[:5]),
            )
            return [r["text"] for r in results]
        except Exception as e:
            logger.error("❌ Retrieval failed: %s — falling back to vector-only", e)
            # Fallback: pure vector search without reranking
            from jet.adapters.llama_cpp.vector_utils import vector_search

            vec_results = vector_search(
                query=query,
                documents=self.documents,
                top_n=self.MAX_CONTEXT_CHUNKS,
            )
            logger.info("🔄 Vector-only fallback: %d results", len(vec_results))
            return [r["text"] for r in vec_results]

    async def _generate(self, query: str, contexts: list[str]) -> str:
        """Real generation via llm_utils.achat with context truncation."""
        try:
            ctx_info = get_model_ctx_embd_size(self.gen_model)
            max_ctx_tokens = ctx_info["ctx"]
        except Exception as e:
            logger.warning(
                "⚠️ Could not get ctx size for %s: %s — using default 4096",
                self.gen_model,
                e,
            )
            max_ctx_tokens = 4096

        reserved_tokens = 1024
        context_budget = max_ctx_tokens - reserved_tokens
        logger.debug(
            "✂️ Context budget: %d tokens (model_ctx=%d, reserved=%d)",
            context_budget,
            max_ctx_tokens,
            reserved_tokens,
        )

        truncated_contexts = truncate_texts(
            contexts,
            model=self.gen_model,
            max_tokens=context_budget,
            strict_sentences=True,
        )
        logger.debug(
            "✂️ Truncated %d contexts → %d chars total",
            len(truncated_contexts),
            sum(len(c) for c in truncated_contexts),
        )

        context_block = "\n\n".join(truncated_contexts)
        messages = [
            {
                "role": "system",
                "content": (
                    "Answer the user's question using ONLY the provided context. "
                    "If the context doesn't contain enough information, say so. "
                    "Do not fabricate information."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context_block}\n\nQuestion: {query}",
            },
        ]

        logger.debug("🤖 Calling generation model=%s", self.gen_model)
        result = await achat(
            prompt_or_messages=messages,
            model=self.gen_model,
            project_name="rag-generation",
            temperature=0.3,
            max_tokens=reserved_tokens,
            enable_thinking=False,
            capture_content=True,
        )
        logger.info(
            "✅ Generation complete: %d chars, finish=%s, tokens=%s",
            len(result.content),
            result.finish_reason,
            result.usage,
        )
        return result.content

    async def _eval_worker(self):
        logger.info("👷 Eval worker running")
        while True:
            try:
                query, contexts, response = await self._eval_queue.get()
                logger.debug("🔧 Eval worker processing: query=%r", query[:60])
                result = await self.evaluator.evaluate_production_async(
                    query,
                    contexts,
                    response,
                )
                self._export_to_observability(result)
                self._eval_queue.task_done()
            except asyncio.CancelledError:
                logger.info("👷 Eval worker cancelled")
                break
            except Exception:
                logger.exception("❌ Production eval worker error")

    def _export_to_observability(self, result):
        status = "🚨 CRITICAL" if result.has_critical_failure else "✅ OK"
        logger.info(
            "%s Eval: stage=%s faith=%.3f halluc=%.3f relevancy=%.3f tokens=%d",
            status,
            result.stage.value,
            result.faithfulness or -1,
            result.hallucination_rate or -1,
            result.answer_relevancy or -1,
            result.total_eval_tokens,
        )
