# jet/adapters/llama_cpp/tasks/rag/eval/service.py
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

        # Pre-compute doc embeddings once for hybrid_search reuse
        from jet.adapters.llama_cpp.embed_utils import embed

        logger.info("Pre-computing document embeddings for %d docs...", len(documents))
        self._doc_embeddings = embed(documents, return_format="numpy")
        logger.info("Document embeddings ready (shape=%s)", self._doc_embeddings.shape)

    async def start(self):
        self._eval_worker_task = asyncio.create_task(self._eval_worker())
        logger.info("RAG evaluation background worker started")

    async def stop(self):
        if self._eval_worker_task:
            self._eval_worker_task.cancel()
            try:
                await self._eval_worker_task
            except asyncio.CancelledError:
                pass

    async def query(self, user_query: str) -> dict:
        # Step 1: Retrieve via hybrid search (vector + reranker)
        contexts = await self._retrieve(user_query)

        # Step 2: Pre-generation gate
        gate = await self.evaluator.evaluate_pre_generation_gate(user_query, contexts)
        if not gate.passed_gate:
            return {
                "answer": "I couldn't find sufficient information to answer reliably.",
                "confidence": "low",
                "debug": {"gate_failed": True, "precision": gate.contextual_precision},
            }

        # Step 3: Generate with context truncation safety
        response = await self._generate(user_query, contexts)

        # Step 4: Queue async production eval
        await self._eval_queue.put((user_query, contexts, response))
        return {"answer": response, "confidence": "high"}

    async def _retrieve(self, query: str) -> list[str]:
        """Real retrieval via hybrid_search with pre-computed embeddings."""
        results = hybrid_search(
            query=query,
            documents=self.documents,
            top_n=self.MAX_CONTEXT_CHUNKS,
            doc_embeddings=self._doc_embeddings,
            normalize_scores=True,
        )
        return [r["text"] for r in results]

    async def _generate(self, query: str, contexts: list[str]) -> str:
        """Real generation via llm_utils.achat with context truncation."""
        # Truncate contexts to fit model context window safely
        try:
            ctx_info = get_model_ctx_embd_size(self.gen_model)
            max_ctx_tokens = ctx_info["ctx"]
        except Exception:
            max_ctx_tokens = 4096

        # Reserve tokens for prompt template + response
        reserved_tokens = 1024
        context_budget = max_ctx_tokens - reserved_tokens
        truncated_contexts = truncate_texts(
            contexts,
            model=self.gen_model,
            max_tokens=context_budget,
            strict_sentences=True,
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

        result = await achat(
            prompt_or_messages=messages,
            model=self.gen_model,
            project_name="rag-generation",
            temperature=0.3,
            max_tokens=reserved_tokens,
            enable_thinking=False,
            capture_content=True,
        )
        return result.content

    async def _eval_worker(self):
        while True:
            try:
                query, contexts, response = await self._eval_queue.get()
                result = await self.evaluator.evaluate_production_async(
                    query, contexts, response
                )
                self._export_to_observability(result)
                self._eval_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Production eval worker error")

    def _export_to_observability(self, result):
        logger.info(
            "Eval: stage=%s faith=%.3f halluc=%.3f relevancy=%.3f tokens=%d",
            result.stage.value,
            result.faithfulness or -1,
            result.hallucination_rate or -1,
            result.answer_relevancy or -1,
            result.total_eval_tokens,
        )
