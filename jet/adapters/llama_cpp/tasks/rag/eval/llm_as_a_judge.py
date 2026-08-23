"""
Production RAG Evaluation Pipeline - Jet LLM Utils Edition
===========================================================
Uses jet.adapters.llama_cpp.llm_utils.achat as the sole LLM backend.
Leverages StreamCompletionResult for structured output, token tracking,
and truncation detection. No external eval library dependencies.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from jet.adapters.llama_cpp.llm_utils import achat
from jet.libs.llama_cpp.usage.chat_stream_types import StreamCompletionResult
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Structured Output Schemas for Judge Tasks
# ---------------------------------------------------------------------------


class RelevanceJudgment(BaseModel):
    """Binary relevance classification for a single retrieved chunk."""

    is_relevant: bool = Field(description="Whether the chunk is relevant to the query")
    reason: str = Field(description="Brief justification for the classification")


class ClaimVerification(BaseModel):
    """Verification of a single claim against retrieved context."""

    claim: str = Field(description="The extracted claim text")
    status: str = Field(description="One of: supported, contradicted, not_mentioned")
    evidence: str = Field(description="Supporting context span or 'none'")


# ---------------------------------------------------------------------------
# 2. Evaluation Result & Stage Definitions
# ---------------------------------------------------------------------------


class EvalStage(Enum):
    OFFLINE_BENCHMARK = "offline_benchmark"
    PRE_GENERATION_GATE = "pre_generation_gate"
    PRODUCTION_ASYNC = "production_async"


@dataclass
class RAGEvaluationResult:
    """Unified result interface for all evaluation stages."""

    stage: EvalStage
    query: str
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    contextual_precision: Optional[float] = None
    contextual_recall: Optional[float] = None
    hallucination_rate: Optional[float] = None
    passed_gate: bool = True
    total_eval_tokens: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def has_critical_failure(self) -> bool:
        if self.faithfulness is not None and self.faithfulness < 0.5:
            return True
        if self.hallucination_rate is not None and self.hallucination_rate > 0.5:
            return True
        if self.contextual_precision is not None and self.contextual_precision < 0.3:
            return True
        return False


# ---------------------------------------------------------------------------
# 3. Jet LLM Judge Adapter
# ---------------------------------------------------------------------------


class JetLLMJudge:
    """
    Central LLM judge using llm_utils.achat exclusively.
    All calls inherit Phoenix tracing, structured output validation,
    and token usage tracking from StreamCompletionResult.
    """

    EVAL_TEMPERATURE = 0.0
    EVAL_MAX_TOKENS = 1024
    CLAIM_MAX_TOKENS = 2048
    EVAL_MODEL = "qwen3.5-uncensored:2b"

    def __init__(self, model: str | None = None, project_prefix: str = "rag-eval"):
        self.model = model or self.EVAL_MODEL
        self.project_prefix = project_prefix

    async def _call_judge(
        self,
        messages: list[dict[str, Any]],
        response_format: Any,
        metric_name: str,
        max_tokens: int | None = None,
    ) -> StreamCompletionResult:
        """Centralized achat call with full StreamCompletionResult handling."""
        result: StreamCompletionResult = await achat(
            prompt_or_messages=messages,
            model=self.model,
            project_name=f"{self.project_prefix}-{metric_name}",
            temperature=self.EVAL_TEMPERATURE,
            max_tokens=max_tokens or self.EVAL_MAX_TOKENS,
            response_format=response_format,
            enable_thinking=False,
            capture_content=True,
        )

        # Track token usage for cost monitoring
        tokens = 0
        if result.usage:
            tokens = result.usage.get("total_tokens", 0)
            logger.debug(
                "Eval [%s] tokens: prompt=%d completion=%d total=%d",
                metric_name,
                result.usage.get("prompt_tokens", 0),
                result.usage.get("completion_tokens", 0),
                tokens,
            )

        # Detect truncation that may corrupt structured output
        if result.finish_reason == "length":
            logger.warning(
                "Judge [%s] truncated at max_tokens=%d. "
                "Structured output may be incomplete; consider increasing.",
                metric_name,
                max_tokens or self.EVAL_MAX_TOKENS,
            )

        return result

    # -- Primitive judge operations -----------------------------------------

    async def extract_claims(self, text: str) -> tuple[list[str], int]:
        """Decompose text into atomic factual claims. Returns (claims, tokens)."""
        messages = [
            {
                "role": "system",
                "content": (
                    "Extract all discrete factual claims from the text. "
                    "Each claim must be independently verifiable. "
                    "Return a JSON array of strings only."
                ),
            },
            {"role": "user", "content": text},
        ]
        result = await self._call_judge(
            messages,
            {"type": "array", "items": {"type": "string"}},
            "claim-extraction",
            max_tokens=self.CLAIM_MAX_TOKENS,
        )
        tokens = result.usage.get("total_tokens", 0) if result.usage else 0
        if not result.structured or not result.structured.success:
            logger.error(
                "Claim extraction failed: %s",
                result.structured.error
                if result.structured
                else "No structured output",
            )
            return [], tokens
        return result.structured.parsed, tokens

    async def judge_chunk_relevance(
        self, query: str, chunk: str
    ) -> tuple[RelevanceJudgment, int]:
        """Classify a single chunk as relevant or not. Returns (judgment, tokens)."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a retrieval relevance judge. Determine if the context "
                    "chunk contains information useful for answering the query. "
                    "Respond with valid JSON matching the schema only."
                ),
            },
            {"role": "user", "content": f"Query: {query}\n\nContext Chunk: {chunk}"},
        ]
        result = await self._call_judge(messages, RelevanceJudgment, "relevance")
        tokens = result.usage.get("total_tokens", 0) if result.usage else 0
        if not result.structured or not result.structured.success:
            error = (
                result.structured.error if result.structured else "No structured output"
            )
            logger.error("Relevance judge failed: %s", error)
            return RelevanceJudgment(
                is_relevant=False, reason=f"Error: {error}"
            ), tokens
        return result.structured.parsed, tokens

    async def verify_claims(
        self, claims: list[str], context: str
    ) -> tuple[list[ClaimVerification], int]:
        """Verify multiple claims against context in one batched call."""
        if not claims:
            return [], 0
        claims_text = "\n".join(f"- {c}" for c in claims)
        messages = [
            {
                "role": "system",
                "content": (
                    "For each claim, determine if it is supported, contradicted, "
                    "or not mentioned in the provided context. "
                    "Return a JSON array of objects matching the schema."
                ),
            },
            {
                "role": "user",
                "content": f"Claims:\n{claims_text}\n\nContext:\n{context}",
            },
        ]
        result = await self._call_judge(
            messages,
            {"type": "array", "items": ClaimVerification.model_json_schema()},
            "faithfulness",
            max_tokens=self.CLAIM_MAX_TOKENS,
        )
        tokens = result.usage.get("total_tokens", 0) if result.usage else 0
        if not result.structured or not result.structured.success:
            logger.error(
                "Claim verification failed: %s",
                result.structured.error
                if result.structured
                else "No structured output",
            )
            return [], tokens
        return result.structured.parsed, tokens

    async def generate_reverse_questions(
        self, answer: str, n: int = 3
    ) -> tuple[list[str], int]:
        """Generate N questions the answer would respond to."""
        messages = [
            {
                "role": "system",
                "content": (
                    f"Generate exactly {n} distinct questions that the given answer "
                    "would be a direct and complete response to. "
                    "Return a JSON array of strings only."
                ),
            },
            {"role": "user", "content": f"Answer: {answer}"},
        ]
        result = await self._call_judge(
            messages,
            {"type": "array", "items": {"type": "string"}},
            "answer-relevancy",
        )
        tokens = result.usage.get("total_tokens", 0) if result.usage else 0
        if not result.structured or not result.structured.success:
            logger.error(
                "Reverse question gen failed: %s",
                result.structured.error
                if result.structured
                else "No structured output",
            )
            return [], tokens
        return result.structured.parsed, tokens


# ---------------------------------------------------------------------------
# 4. Metric Computation Layer
# ---------------------------------------------------------------------------


class RAGMetrics:
    """
    Pure metric computation using JetLLMJudge primitives.
    No external eval library dependencies.
    """

    CONTEXT_PRECISION_THRESHOLD = 0.5
    FAITHFULNESS_THRESHOLD = 0.7
    ANSWER_RELEVANCY_THRESHOLD = 0.6
    HALLUCINATION_THRESHOLD = 0.5

    def __init__(self, judge: JetLLMJudge):
        self.judge = judge

    async def compute_contextual_precision(
        self, query: str, contexts: list[str]
    ) -> tuple[float, int]:
        """Fraction of top-ranked chunks that are relevant (position-weighted)."""
        if not contexts:
            return 0.0, 0

        judgments = await asyncio.gather(
            *[self.judge.judge_chunk_relevance(query, chunk) for chunk in contexts]
        )
        total_tokens = sum(tokens for _, tokens in judgments)

        # Position-weighted precision: relevant chunks ranked higher score more
        weighted_score = 0.0
        relevant_count = 0
        for i, (judgment, _) in enumerate(judgments):
            if judgment.is_relevant:
                relevant_count += 1
                weighted_score += relevant_count / (i + 1)

        precision = weighted_score / len(contexts) if contexts else 0.0
        return precision, total_tokens

    async def compute_faithfulness(
        self, response: str, contexts: list[str]
    ) -> tuple[float, float, int]:
        """Returns (faithfulness_score, hallucination_rate, total_tokens)."""
        context_text = "\n---\n".join(contexts)

        # Step 1: Extract claims from response
        claims, extract_tokens = await self.judge.extract_claims(response)
        if not claims:
            return 1.0, 0.0, extract_tokens  # No claims = vacuously faithful

        # Step 2: Verify all claims against context in one batched call
        verifications, verify_tokens = await self.judge.verify_claims(
            claims, context_text
        )
        total_tokens = extract_tokens + verify_tokens

        if not verifications:
            return 0.0, 1.0, total_tokens

        supported = sum(1 for v in verifications if v.status == "supported")
        contradicted = sum(1 for v in verifications if v.status == "contradicted")
        not_mentioned = sum(1 for v in verifications if v.status == "not_mentioned")

        faithfulness = supported / len(verifications)
        hallucination_rate = (contradicted + not_mentioned) / len(verifications)

        return faithfulness, hallucination_rate, total_tokens

    async def compute_answer_relevancy(
        self, query: str, response: str
    ) -> tuple[float, int]:
        """Semantic similarity between query and reverse-generated questions."""
        questions, tokens = await self.judge.generate_reverse_questions(response)
        if not questions:
            return 0.0, tokens

        # Simple lexical overlap as lightweight proxy (avoids extra embedding call)
        # In production, replace with cosine similarity over embeddings
        query_tokens = set(query.lower().split())
        similarities = []
        for q in questions:
            q_tokens = set(q.lower().split())
            overlap = len(query_tokens & q_tokens) / max(
                len(query_tokens | q_tokens), 1
            )
            similarities.append(overlap)

        relevancy = sum(similarities) / len(similarities)
        return relevancy, tokens


# ---------------------------------------------------------------------------
# 5. Evaluator Engine
# ---------------------------------------------------------------------------


class RAGEvaluator:
    """Orchestrates metric computation across evaluation stages."""

    def __init__(self, model: str | None = None):
        self.judge = JetLLMJudge(model=model)
        self.metrics = RAGMetrics(self.judge)

    async def evaluate_pre_generation_gate(
        self, query: str, contexts: list[str]
    ) -> RAGEvaluationResult:
        """Fast sync gate before generation. Blocks on critically bad retrieval."""
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

    async def evaluate_production_async(
        self, query: str, contexts: list[str], response: str
    ) -> RAGEvaluationResult:
        """Reference-free safety eval. Runs after response is sent to user."""
        # Run faithfulness and answer relevancy concurrently
        (
            (faithfulness, halluc_rate, faith_tokens),
            (relevancy, rel_tokens),
        ) = await asyncio.gather(
            self.metrics.compute_faithfulness(response, contexts),
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

    async def evaluate_offline(
        self, query: str, contexts: list[str], response: str, reference: str
    ) -> RAGEvaluationResult:
        """Full benchmark suite. Requires ground-truth reference for recall."""
        # Note: Contextual Recall requires comparing claims in reference vs.
        # attributability to context. Reuses verify_claims with reference claims.
        ref_claims, ref_tokens = await self.judge.extract_claims(reference)
        context_text = "\n---\n".join(contexts)
        ref_verifications, ref_verify_tokens = await self.judge.verify_claims(
            ref_claims, context_text
        )
        recall_tokens = ref_tokens + ref_verify_tokens

        attributable = sum(1 for v in ref_verifications if v.status == "supported")
        contextual_recall = (
            attributable / len(ref_verifications) if ref_verifications else 0.0
        )

        # Also run precision, faithfulness, and relevancy
        precision, prec_tokens = await self.metrics.compute_contextual_precision(
            query, contexts
        )
        (
            (faithfulness, halluc_rate, faith_tokens),
            (relevancy, rel_tokens),
        ) = await asyncio.gather(
            self.metrics.compute_faithfulness(response, contexts),
            self.metrics.compute_answer_relevancy(query, response),
        )

        return RAGEvaluationResult(
            stage=EvalStage.OFFLINE_BENCHMARK,
            query=query,
            contextual_precision=precision,
            contextual_recall=contextual_recall,
            faithfulness=faithfulness,
            hallucination_rate=halluc_rate,
            answer_relevancy=relevancy,
            total_eval_tokens=prec_tokens + recall_tokens + faith_tokens + rel_tokens,
            metadata={
                "reference_claims_count": len(ref_claims),
                "attributable_claims": attributable,
            },
        )


# ---------------------------------------------------------------------------
# 6. RAG Service Integration
# ---------------------------------------------------------------------------


class RAGService:
    """
    Production RAG service with three-stage evaluation.
    Pre-gen gate is synchronous; production eval is fully async.
    """

    def __init__(self, evaluator: RAGEvaluator):
        self.evaluator = evaluator
        self._eval_queue: asyncio.Queue = asyncio.Queue()
        self._eval_worker_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start background worker for non-blocking production evaluation."""
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
        # Step 1: Retrieve
        contexts = await self._retrieve(user_query)

        # Step 2: PRE-GENERATION GATE (sync, blocks on critical failure)
        gate = await self.evaluator.evaluate_pre_generation_gate(user_query, contexts)

        if not gate.passed_gate:
            return {
                "answer": "I couldn't find sufficient information to answer reliably.",
                "confidence": "low",
                "debug": {"gate_failed": True, "precision": gate.contextual_precision},
            }

        # Step 3: Generate
        response = await self._generate(user_query, contexts)

        # Step 4: Queue PRODUCTION ASYNC eval (never blocks user response)
        await self._eval_queue.put((user_query, contexts, response))

        return {"answer": response, "confidence": "high"}

    async def _eval_worker(self):
        """Background worker processing production evals off critical path."""
        while True:
            try:
                query, contexts, response = await self._eval_queue.get()
                result = await self.evaluator.evaluate_production_async(
                    query, contexts, response
                )
                self._export_to_observability(result)
                self._eval_queue.task_done()
            except asyncio.CancelledError:
                logger.info("Eval worker cancelled")
                break
            except Exception:
                logger.exception("Production eval worker error")

    def _export_to_observability(self, result: RAGEvaluationResult):
        """Push metrics to Phoenix/dashboard. Token count enables cost tracking."""
        logger.info(
            "Eval exported: stage=%s faith=%.3f halluc=%.3f relevancy=%.3f tokens=%d",
            result.stage.value,
            result.faithfulness or -1,
            result.hallucination_rate or -1,
            result.answer_relevancy or -1,
            result.total_eval_tokens,
        )
        # TODO: Push to Phoenix custom metrics, Prometheus, or Datadog

    # -- Placeholder infrastructure methods ---------------------------------

    async def _retrieve(self, query: str) -> list[str]:
        """Replace with your actual vector store + reranker call."""
        raise NotImplementedError("Implement your retrieval pipeline")

    async def _generate(self, query: str, contexts: list[str]) -> str:
        """Replace with your actual LLM generation call."""
        raise NotImplementedError("Implement your generation pipeline")
