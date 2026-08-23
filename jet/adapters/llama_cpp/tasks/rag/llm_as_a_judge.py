"""
Production RAG Evaluation Pipeline - DeepEval Edition

Drop-in replacement for RAGAS version. Uses DeepEval's LLMTestCase
and metric classes with async background processing.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    HallucinationMetric,
)
from deepeval.models import GPTModel
from deepeval.test_case import LLMTestCase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Configuration & Data Models (unchanged interface)
# ---------------------------------------------------------------------------


class EvalStage(Enum):
    OFFLINE_BENCHMARK = "offline_benchmark"
    PRE_GENERATION_GATE = "pre_generation_gate"
    PRODUCTION_ASYNC = "production_async"


@dataclass
class RAGEvaluationResult:
    """Same interface as RAGAS version for drop-in compatibility."""

    stage: EvalStage
    query: str
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    contextual_precision: Optional[float] = None
    contextual_recall: Optional[float] = None
    hallucination_score: Optional[float] = None  # DeepEval bonus metric
    passed_gate: bool = True
    metadata: dict = field(default_factory=dict)

    @property
    def has_critical_failure(self) -> bool:
        if self.faithfulness is not None and self.faithfulness < 0.5:
            return True
        if self.hallucination_score is not None and self.hallucination_score > 0.5:
            return True
        if self.contextual_precision is not None and self.contextual_precision < 0.3:
            return True
        return False


# ---------------------------------------------------------------------------
# 2. DeepEval Evaluator Engine
# ---------------------------------------------------------------------------


class DeepEvalRAGEvaluator:
    """
    Central evaluator using DeepEval metrics.
    Key difference from RAGAS: metrics are instantiated per-evaluation
    with thresholds baked in, enabling automatic pass/fail semantics.
    """

    FAITHFULNESS_THRESHOLD = 0.7
    CONTEXT_PRECISION_THRESHOLD = 0.5
    ANSWER_RELEVANCY_THRESHOLD = 0.6
    HALLUCINATION_THRESHOLD = 0.5  # Lower is better in DeepEval

    def __init__(self, model: str = "gpt-4o-mini"):
        self._model = GPTModel(model=model)
        logger.info("DeepEvalRAGEvaluator initialized with model=%s", model)

    def _build_test_case(
        self,
        query: str,
        contexts: list[str],
        response: Optional[str] = None,
        reference: Optional[str] = None,
    ) -> LLMTestCase:
        """DeepEval requires LLMTestCase objects instead of raw dicts."""
        return LLMTestCase(
            input=query,
            actual_output=response or "",
            retrieval_context=contexts,
            expected_output=reference,  # Only used in offline benchmark
        )

    # -- Stage-specific evaluation methods ----------------------------------

    async def evaluate_offline(
        self, query: str, contexts: list[str], response: str, reference: str
    ) -> RAGEvaluationResult:
        """Full benchmark with ground truth. Includes ContextRecall."""
        tc = self._build_test_case(query, contexts, response, reference)

        metrics = [
            FaithfulnessMetric(
                threshold=self.FAITHFULNESS_THRESHOLD, model=self._model
            ),
            AnswerRelevancyMetric(
                threshold=self.ANSWER_RELEVANCY_THRESHOLD, model=self._model
            ),
            ContextualPrecisionMetric(
                threshold=self.CONTEXT_PRECISION_THRESHOLD, model=self._model
            ),
            ContextualRecallMetric(threshold=0.7, model=self._model),
        ]

        # DeepEval supports batch async evaluation
        await asyncio.gather(*[m.a_measure(tc) for m in metrics])

        return RAGEvaluationResult(
            stage=EvalStage.OFFLINE_BENCHMARK,
            query=query,
            faithfulness=metrics[0].score,
            answer_relevancy=metrics[1].score,
            contextual_precision=metrics[2].score,
            contextual_recall=metrics[3].score,
            metadata={
                m.__name__: {"pass": m.is_successful(), "reason": m.reason}
                for m in metrics
            },
        )

    async def evaluate_pre_generation_gate(
        self, query: str, contexts: list[str]
    ) -> RAGEvaluationResult:
        """Fast retrieval-only gate before generation."""
        tc = self._build_test_case(query, contexts)
        metric = ContextualPrecisionMetric(
            threshold=self.CONTEXT_PRECISION_THRESHOLD, model=self._model
        )
        await metric.a_measure(tc)

        passed = metric.is_successful()
        if not passed:
            logger.warning(
                "Pre-gen gate FAILED: score=%.3f reason=%s query=%r",
                metric.score,
                metric.reason,
                query[:80],
            )
        return RAGEvaluationResult(
            stage=EvalStage.PRE_GENERATION_GATE,
            query=query,
            contextual_precision=metric.score,
            passed_gate=passed,
            metadata={"reason": metric.reason},
        )

    async def evaluate_production_async(
        self, query: str, contexts: list[str], response: str
    ) -> RAGEvaluationResult:
        """Reference-free safety eval with DeepEval-exclusive HallucinationMetric."""
        tc = self._build_test_case(query, contexts, response)

        faith_metric = FaithfulnessMetric(
            threshold=self.FAITHFULNESS_THRESHOLD, model=self._model
        )
        relevancy_metric = AnswerRelevancyMetric(
            threshold=self.ANSWER_RELEVANCY_THRESHOLD, model=self._model
        )
        halluc_metric = HallucinationMetric(
            threshold=self.HALLUCINATION_THRESHOLD, model=self._model
        )

        await asyncio.gather(
            faith_metric.a_measure(tc),
            relevancy_metric.a_measure(tc),
            halluc_metric.a_measure(tc),
        )

        result = RAGEvaluationResult(
            stage=EvalStage.PRODUCTION_ASYNC,
            query=query,
            faithfulness=faith_metric.score,
            answer_relevancy=relevancy_metric.score,
            hallucination_score=halluc_metric.score,
            metadata={
                "faithfulness_reason": faith_metric.reason,
                "hallucination_reason": halluc_metric.reason,
            },
        )
        if result.has_critical_failure:
            logger.error("CRITICAL RAG FAILURE: %s", result)
        return result


# ---------------------------------------------------------------------------
# 3. RAGService integration (identical structure, swapped evaluator)
# ---------------------------------------------------------------------------


class RAGService:
    """Same service pattern as RAGAS version — only evaluator class changes."""

    def __init__(self, evaluator: DeepEvalRAGEvaluator):
        self.evaluator = evaluator
        self._eval_queue: asyncio.Queue = asyncio.Queue()
        self._eval_worker_task: Optional[asyncio.Task] = None

    async def start(self):
        self._eval_worker_task = asyncio.create_task(self._eval_worker())

    async def stop(self):
        if self._eval_worker_task:
            self._eval_worker_task.cancel()

    async def query(self, user_query: str) -> dict:
        # Retrieve
        contexts = await self._retrieve(user_query)

        # PRE-GEN GATE (sync, blocks on bad retrieval)
        gate = await self.evaluator.evaluate_pre_generation_gate(user_query, contexts)
        if not gate.passed_gate:
            return {"answer": "Insufficient information found.", "confidence": "low"}

        # Generate
        response = await self._generate(user_query, contexts)

        # Queue async production eval (non-blocking)
        await self._eval_queue.put((user_query, contexts, response))

        return {"answer": response, "confidence": "high"}

    async def _eval_worker(self):
        while True:
            try:
                q, c, r = await self._eval_queue.get()
                result = await self.evaluator.evaluate_production_async(q, c, r)
                self._export_to_observability(result)
                self._eval_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Eval worker error")

    async def _retrieve(self, query: str) -> list[str]:
        raise NotImplementedError

    async def _generate(self, query: str, contexts: list[str]) -> str:
        raise NotImplementedError

    def _export_to_observability(self, result: RAGEvaluationResult):
        logger.info(
            "DeepEval export: stage=%s faith=%.3f halluc=%.3f",
            result.stage.value,
            result.faithfulness or -1,
            result.hallucination_score or -1,
        )
