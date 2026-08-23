"""
RAG Evaluator - Stage Orchestration Layer
==========================================
Orchestrates RAGMetrics across three evaluation stages:
1. Pre-generation gate (sync, blocks bad retrieval)
2. Production async eval (reference-free safety monitoring)
3. Offline benchmark (full suite with ground-truth recall)

Uses chunking_utils.truncate_texts to prevent context overflow
during faithfulness/recall verification against large context sets.
"""

from __future__ import annotations

import asyncio
import logging

from jet.adapters.llama_cpp.chunking_utils import truncate_texts
from jet.adapters.llama_cpp.model_utils import get_model_ctx_embd_size

from .judge import JetLLMJudge
from .metrics import RAGMetrics
from .types import EvalStage, RAGEvaluationResult

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    Orchestrates metric computation across evaluation stages.

    All judge calls go through JetLLMJudge → llm_utils.achat,
    inheriting Phoenix tracing, structured output validation,
    and token usage tracking automatically.
    """

    # Reserve tokens for system prompt + response in judge calls
    JUDGE_CONTEXT_RESERVE_TOKENS = 1024

    def __init__(self, model: str | None = None):
        self.judge = JetLLMJudge(model=model)
        self.metrics = RAGMetrics(self.judge)
        self._judge_model = self.judge.model

    def _get_judge_context_budget(self) -> int:
        """Dynamically compute max context tokens for judge verification calls."""
        try:
            ctx_info = get_model_ctx_embd_size(self._judge_model)
            return max(ctx_info["ctx"] - self.JUDGE_CONTEXT_RESERVE_TOKENS, 512)
        except Exception:
            logger.warning(
                "Could not get context size for judge model %s, using default 3072",
                self._judge_model,
            )
            return 3072

    def _truncate_contexts_for_judge(self, contexts: list[str]) -> list[str]:
        """Truncate concatenated context to fit judge model's context window."""
        budget = self._get_judge_context_budget()
        combined = "\n---\n".join(contexts)
        truncated = truncate_texts(
            combined,
            model=self._judge_model,
            max_tokens=budget,
            strict_sentences=True,
            show_progress=False,
        )
        if isinstance(truncated, list):
            truncated = truncated[0] if truncated else ""
        return [truncated] if truncated else contexts

    # ------------------------------------------------------------------
    # Stage 1: Pre-Generation Gate
    # ------------------------------------------------------------------

    async def evaluate_pre_generation_gate(
        self, query: str, contexts: list[str]
    ) -> RAGEvaluationResult:
        """
        Fast synchronous gate before generation.
        Blocks generation if retrieval quality is critically low.

        Only computes Contextual Precision (fastest retrieval-only metric).
        Does NOT call generation or faithfulness — those happen after the gate passes.
        """
        precision, tokens = await self.metrics.compute_contextual_precision(
            query, contexts
        )
        passed = precision >= RAGMetrics.CONTEXT_PRECISION_THRESHOLD

        if not passed:
            logger.warning(
                "Pre-gen gate FAILED: precision=%.3f (threshold=%.2f) query=%r",
                precision,
                RAGMetrics.CONTEXT_PRECISION_THRESHOLD,
                query[:80],
            )

        return RAGEvaluationResult(
            stage=EvalStage.PRE_GENERATION_GATE,
            query=query,
            contextual_precision=precision,
            passed_gate=passed,
            total_eval_tokens=tokens,
        )

    # ------------------------------------------------------------------
    # Stage 2: Production Async Evaluation
    # ------------------------------------------------------------------

    async def evaluate_production_async(
        self, query: str, contexts: list[str], response: str
    ) -> RAGEvaluationResult:
        """
        Reference-free safety evaluation.
        Runs AFTER response is sent to user via background worker.

        Computes faithfulness, hallucination rate, and answer relevancy.
        Context is truncated to fit judge model window before verification.
        """
        # Truncate contexts to prevent judge overflow on large retrievals
        safe_contexts = self._truncate_contexts_for_judge(contexts)

        # Run faithfulness and answer relevancy concurrently
        (
            (faithfulness, halluc_rate, faith_tokens),
            (relevancy, rel_tokens),
        ) = await asyncio.gather(
            self.metrics.compute_faithfulness(response, safe_contexts),
            self.metrics.compute_answer_relevancy(query, response),
        )

        result = RAGEvaluationResult(
            stage=EvalStage.PRODUCTION_ASYNC,
            query=query,
            faithfulness=faithfulness,
            hallucination_rate=halluc_rate,
            answer_relevancy=relevancy,
            total_eval_tokens=faith_tokens + rel_tokens,
        )

        if result.has_critical_failure:
            logger.error(
                "CRITICAL RAG FAILURE: faith=%.3f halluc=%.3f query=%r",
                faithfulness,
                halluc_rate,
                query[:80],
            )

        return result

    # ------------------------------------------------------------------
    # Stage 3: Offline Benchmark
    # ------------------------------------------------------------------

    async def evaluate_offline(
        self,
        query: str,
        contexts: list[str],
        response: str,
        reference: str,
    ) -> RAGEvaluationResult:
        """
        Full benchmark suite with ground-truth reference.
        Used in CI/CD regression testing and model comparison.

        Computes ALL metrics including Contextual Recall (requires reference).
        Context is truncated for both faithfulness and recall verification.
        """
        # Truncate contexts once for both faithfulness and recall
        safe_contexts = self._truncate_contexts_for_judge(contexts)
        context_text = "\n---\n".join(safe_contexts)

        # --- Contextual Recall: can reference claims be attributed to context? ---
        ref_claims, ref_extract_tokens = await self.judge.extract_claims(reference)
        ref_verifications, ref_verify_tokens = await self.judge.verify_claims(
            ref_claims, context_text
        )
        recall_tokens = ref_extract_tokens + ref_verify_tokens

        attributable = sum(1 for v in ref_verifications if v.status == "supported")
        contextual_recall = (
            attributable / len(ref_verifications) if ref_verifications else 0.0
        )

        # --- Remaining metrics (concurrent) ---
        precision, prec_tokens = await self.metrics.compute_contextual_precision(
            query,
            contexts,  # Full contexts OK here — per-chunk, not concatenated
        )

        (
            (faithfulness, halluc_rate, faith_tokens),
            (relevancy, rel_tokens),
        ) = await asyncio.gather(
            self.metrics.compute_faithfulness(response, safe_contexts),
            self.metrics.compute_answer_relevancy(query, response),
        )

        total_tokens = prec_tokens + recall_tokens + faith_tokens + rel_tokens

        return RAGEvaluationResult(
            stage=EvalStage.OFFLINE_BENCHMARK,
            query=query,
            contextual_precision=precision,
            contextual_recall=contextual_recall,
            faithfulness=faithfulness,
            hallucination_rate=halluc_rate,
            answer_relevancy=relevancy,
            total_eval_tokens=total_tokens,
            metadata={
                "reference_claims_count": len(ref_claims),
                "attributable_claims": attributable,
                "judge_context_budget": self._get_judge_context_budget(),
            },
        )
