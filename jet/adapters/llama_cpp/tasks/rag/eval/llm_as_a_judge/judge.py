# jet/adapters/llama_cpp/tasks/rag/eval/judge.py
"""LLM-as-a-Judge primitives using llm_utils.achat."""

from __future__ import annotations

import logging
from typing import Any

from jet.adapters.llama_cpp.llm_utils import achat
from jet.libs.llama_cpp.usage.chat_stream_types import StreamCompletionResult

from .types import ClaimVerification, RelevanceJudgment

logger = logging.getLogger(__name__)


class JetLLMJudge:
    """Central LLM judge using llm_utils.achat exclusively."""

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
        tokens = result.usage.get("total_tokens", 0) if result.usage else 0
        logger.debug(
            "Eval [%s] tokens: p=%d c=%d total=%d",
            metric_name,
            result.usage.get("prompt_tokens", 0) if result.usage else 0,
            result.usage.get("completion_tokens", 0) if result.usage else 0,
            tokens,
        )
        if result.finish_reason == "length":
            logger.warning(
                "Judge [%s] truncated at max_tokens=%d",
                metric_name,
                max_tokens or self.EVAL_MAX_TOKENS,
            )
        return result

    async def extract_claims(self, text: str) -> tuple[list[str], int]:
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
        self,
        query: str,
        chunk: str,
    ) -> tuple[RelevanceJudgment, int]:
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
        self,
        claims: list[str],
        context: str,
    ) -> tuple[list[ClaimVerification], int]:
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
        self,
        answer: str,
        n: int = 3,
    ) -> tuple[list[str], int]:
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
