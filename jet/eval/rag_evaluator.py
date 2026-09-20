from contextlib import contextmanager
from typing import Any, Dict, List, Optional

# DeepEval imports
from deepeval import evaluate
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.evaluate import AsyncConfig, DisplayConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

# OpenTelemetry imports for context management
from opentelemetry import context as otel_context

from jet.adapters.deepeval.factory import get_llm_client

# Reuse existing Jet modules
from jet.logger import logger
from jet.observability import llm_span, redact, reranker_span, tool_span

# Shared sync config to avoid repeated object creation
_SYNC_CONFIG = AsyncConfig(run_async=False)
# Disable DeepEval's internal progress bar to prevent it from covering our logs
_DISPLAY_CONFIG = DisplayConfig(show_indicator=False)


class RAGEvaluator:
    """
    Encapsulates DeepEval logic for the JetScripts RAG pipeline.
    Evaluates chunking, retrieval, reranking, and final generation.

    Features:
    - Uses CONFIDENT_BASE_URL env var for self-hosted Phoenix/Confident.
    - Uses jet.adapters.deepeval for consistent LLM-as-a-judge behavior.
    - Preserves OTel trace context so eval spans nest under the main pipeline trace.
    """

    def __init__(
        self,
        project_name: str = "jet-rag-evals",
        llm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.project_name = project_name

        # Initialize standard RAG metrics using your custom llama.cpp adapter
        llm_client = get_llm_client(**(llm_kwargs or {}))

        self.faithfulness = FaithfulnessMetric(model=llm_client)
        self.answer_relevancy = AnswerRelevancyMetric(model=llm_client)
        self.contextual_relevancy = ContextualRelevancyMetric(model=llm_client)
        self.contextual_precision = ContextualPrecisionMetric(model=llm_client)
        self.contextual_recall = ContextualRecallMetric(model=llm_client)

        # Custom metric for context coherence using G-Eval
        self.context_coherence = GEval(
            name="Context Coherence",
            criteria=(
                "Determine if the assembled context flows logically. "
                "If headers are present, check hierarchy. "
                "If not, check for disjointed sentences or abrupt cuts in the narrative."
            ),
            evaluation_params=[LLMTestCaseParams.RETRIEVAL_CONTEXT],
            model=llm_client,
        )

    @contextmanager
    def _preserve_trace_context(self):
        """
        Context manager to attach the current OTel trace context to DeepEval calls.
        This ensures all judge LLM calls appear as children of the active pipeline span.
        """
        current_ctx = otel_context.get_current()
        token = otel_context.attach(current_ctx)
        try:
            yield
        finally:
            otel_context.detach(token)

    def _get_metric_score(self, test_result, metric_name: str) -> float:
        """Helper to safely extract a score from TestResult.metrics_data."""
        if not test_result.metrics_data:
            return 0.0
        for metric_data in test_result.metrics_data:
            if metric_data.name == metric_name:
                return metric_data.score
        return 0.0

    def _get_metric_reason(self, test_result, metric_name: str) -> str:
        """Helper to safely extract a reason from TestResult.metrics_data."""
        if not test_result.metrics_data:
            return "No reason provided"
        for metric_data in test_result.metrics_data:
            if metric_data.name == metric_name:
                return metric_data.reason or "No reason provided"
        return "No reason provided"

    def evaluate_chunk_quality(self, chunks: List[str], query: str) -> Dict[str, Any]:
        """Evaluate if chunks are semantically complete and relevant."""
        test_cases = [
            LLMTestCase(input=query, actual_output=chunk, retrieval_context=[chunk])
            for chunk in chunks
        ]

        with self._preserve_trace_context():
            with tool_span(
                name="eval.chunk_quality",
                tool_name="contextual_relevancy",
                parameters={"query": query, "chunk_count": len(chunks)},
            ) as span:
                try:
                    results = evaluate(
                        test_cases=test_cases,
                        metrics=[self.contextual_relevancy],
                        async_config=_SYNC_CONFIG,
                        display_config=_DISPLAY_CONFIG,
                    )
                    avg_score = sum(
                        self._get_metric_score(r, "Contextual Relevancy")
                        for r in results.test_results
                    ) / max(len(results.test_results), 1)

                    span.set_attribute("eval.average_score", avg_score)
                    return {
                        "metric": "Contextual Relevancy",
                        "average_score": avg_score,
                    }
                except Exception as e:
                    logger.error(f"Chunk evaluation failed: {e}")
                    span.record_exception(e)
                    return {"error": str(e)}

    def evaluate_retrieval_ranking(
        self,
        query: str,
        retrieved_chunks: List[str],
        expected_relevant_chunks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Evaluate the precision and recall of the vector search/reranking step."""
        if not retrieved_chunks:
            return {"error": "No chunks retrieved"}

        with self._preserve_trace_context():
            with reranker_span(
                name="eval.retrieval_ranking",
                model_name="deepeval-metrics",
                query=query,
                documents=retrieved_chunks,
                top_k=len(retrieved_chunks),
            ) as span:
                if expected_relevant_chunks:
                    test_case = LLMTestCase(
                        input=query,
                        actual_output="",
                        retrieval_context=retrieved_chunks,
                        expected_output="\n".join(expected_relevant_chunks),
                    )
                    try:
                        results = evaluate(
                            test_cases=[test_case],
                            metrics=[self.contextual_precision, self.contextual_recall],
                            async_config=_SYNC_CONFIG,
                            display_config=_DISPLAY_CONFIG,
                        )
                        res = results.test_results[0]
                        precision = self._get_metric_score(res, "Contextual Precision")
                        recall = self._get_metric_score(res, "Contextual Recall")

                        span.set_attribute("eval.precision", precision)
                        span.set_attribute("eval.recall", recall)

                        return {
                            "metric": "Retrieval Precision/Recall",
                            "precision": precision,
                            "recall": recall,
                        }
                    except Exception as e:
                        logger.error(f"Retrieval evaluation failed: {e}")
                        span.record_exception(e)
                        return {"error": str(e)}
                else:
                    logger.warning(
                        "No expected output provided. Skipping Precision/Recall."
                    )
                    return {"skipped": True, "reason": "No ground truth"}

    def evaluate_final_answer(
        self, query: str, answer: str, context: str
    ) -> Dict[str, Any]:
        """Evaluate the final LLM response for faithfulness and relevancy."""
        test_case = LLMTestCase(
            input=query, actual_output=answer, retrieval_context=[context]
        )

        with self._preserve_trace_context():
            with llm_span(
                name="eval.final_answer",
                model_name="deepeval-judge",
                messages=[
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": answer},
                ],
                invocation_params={
                    "metrics": ["faithfulness", "relevancy", "coherence"]
                },
            ) as span:
                try:
                    results = evaluate(
                        test_cases=[test_case],
                        metrics=[
                            self.faithfulness,
                            self.answer_relevancy,
                            self.context_coherence,
                        ],
                        async_config=_SYNC_CONFIG,
                        display_config=_DISPLAY_CONFIG,
                    )
                    res = results.test_results[0]

                    faithfulness_score = self._get_metric_score(res, "Faithfulness")
                    relevancy_score = self._get_metric_score(res, "Answer Relevancy")
                    coherence_score = self._get_metric_score(res, "Context Coherence")

                    span.set_attribute("eval.faithfulness", faithfulness_score)
                    span.set_attribute("eval.answer_relevancy", relevancy_score)
                    span.set_attribute("eval.context_coherence", coherence_score)

                    return {
                        "faithfulness": faithfulness_score,
                        "answer_relevancy": relevancy_score,
                        "context_coherence": coherence_score,
                        "reasons": {
                            "faithfulness": self._get_metric_reason(
                                res, "Faithfulness"
                            ),
                            "relevancy": self._get_metric_reason(
                                res, "Answer Relevancy"
                            ),
                            "coherence": self._get_metric_reason(
                                res, "Context Coherence"
                            ),
                        },
                    }
                except Exception as e:
                    logger.error(f"Final answer evaluation failed: {e}")
                    span.record_exception(e)
                    return {"error": str(e)}

    def run_pipeline_evaluation(
        self,
        query: str,
        retrieved_chunks: List[str],
        final_answer: str,
        expected_answer: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a comprehensive evaluation of the entire RAG flow."""
        context = "\n\n".join(retrieved_chunks)
        return {
            "query": redact(query),
            "retrieval": self.evaluate_retrieval_ranking(query, retrieved_chunks),
            "generation": self.evaluate_final_answer(query, final_answer, context),
        }


def create_golden_dataset(queries: List[Dict[str, Any]]) -> EvaluationDataset:
    """Create a dataset from a list of dicts with 'input' and optional 'expected_output'."""
    goldens = [
        Golden(input=item["input"], expected_output=item.get("expected_output"))
        for item in queries
    ]
    return EvaluationDataset(goldens=goldens)
