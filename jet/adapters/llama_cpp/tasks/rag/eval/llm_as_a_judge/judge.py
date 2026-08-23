# jet/adapters/llama_cpp/tasks/rag/eval/llm_as_a_judge/judge.py
"""LLM-as-a-Judge primitives using llm_utils.achat.

NOTE: All judge prompts use a SINGLE user message (no system message).
This avoids llm_utils injecting a schema system prompt at index 1,
which breaks Qwen3.5's Jinja template ("System message must be at the beginning").
Schema adherence is enforced via response_format + inline instructions.
"""

from __future__ import annotations

import json
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
        logger.debug(
            "🔍 Judge [%s] calling achat: model=%s, msgs=%d, format=%s",
            metric_name,
            self.model,
            len(messages),
            type(response_format).__name__,
        )
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
            "✅ Judge [%s] completed: tokens=%d, finish=%s, structured=%s",
            metric_name,
            tokens,
            result.finish_reason,
            result.structured.success if result.structured else "None",
        )
        if result.finish_reason == "length":
            logger.warning(
                "⚠️ Judge [%s] truncated at max_tokens=%d",
                metric_name,
                max_tokens or self.EVAL_MAX_TOKENS,
            )
        return result

    async def extract_claims(self, text: str) -> tuple[list[str], int]:
        # Single user message — no system message to avoid template conflict
        messages = [
            {
                "role": "user",
                "content": (
                    "Extract all discrete factual claims from the following text. "
                    "Each claim must be independently verifiable. "
                    "Return ONLY a JSON array of strings.\n\n"
                    f"Text: {text}"
                ),
            },
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
                "❌ Claim extraction failed: %s | raw_content=%r",
                result.structured.error
                if result.structured
                else "No structured output",
                result.content[:200],
            )
            return [], tokens
        logger.debug("📝 Extracted %d claims", len(result.structured.parsed))
        return result.structured.parsed, tokens

    async def judge_chunk_relevance(
        self,
        query: str,
        chunk: str,
    ) -> tuple[RelevanceJudgment, int]:
        # Single user message with inline schema description
        schema_desc = json.dumps(RelevanceJudgment.model_json_schema(), indent=2)
        messages = [
            {
                "role": "user",
                "content": (
                    "You are a retrieval relevance judge. Determine if the context "
                    "chunk contains information useful for answering the query.\n\n"
                    f"Query: {query}\n\nContext Chunk: {chunk}\n\n"
                    f"Respond with valid JSON matching this schema:\n{schema_desc}"
                ),
            },
        ]
        result = await self._call_judge(messages, RelevanceJudgment, "relevance")
        tokens = result.usage.get("total_tokens", 0) if result.usage else 0
        if not result.structured or not result.structured.success:
            error = (
                result.structured.error if result.structured else "No structured output"
            )
            logger.error(
                "❌ Relevance judge failed: %s | raw=%r", error, result.content[:200]
            )
            return RelevanceJudgment(
                is_relevant=False, reason=f"Error: {error}"
            ), tokens
        logger.debug(
            "🏷️ Chunk relevance: is_relevant=%s",
            result.structured.parsed.is_relevant,
        )
        return result.structured.parsed, tokens

    async def verify_claims(
        self,
        claims: list[str],
        context: str,
    ) -> tuple[list[ClaimVerification], int]:
        if not claims:
            return [], 0
        claims_text = "\n".join(f"- {c}" for c in claims)
        schema_desc = json.dumps(ClaimVerification.model_json_schema(), indent=2)
        messages = [
            {
                "role": "user",
                "content": (
                    "For each claim below, determine if it is supported, contradicted, "
                    "or not mentioned in the provided context.\n\n"
                    f"Claims:\n{claims_text}\n\nContext:\n{context}\n\n"
                    f"Return a JSON array of objects matching this schema:\n{schema_desc}"
                ),
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
                "❌ Claim verification failed: %s | raw=%r",
                result.structured.error
                if result.structured
                else "No structured output",
                result.content[:200],
            )
            return [], tokens
        logger.debug("🔎 Verified %d claims", len(result.structured.parsed))
        return result.structured.parsed, tokens

    async def generate_reverse_questions(
        self,
        answer: str,
        n: int = 3,
    ) -> tuple[list[str], int]:
        messages = [
            {
                "role": "user",
                "content": (
                    f"Generate exactly {n} distinct questions that the given answer "
                    "would be a direct and complete response to. "
                    "Return ONLY a JSON array of strings.\n\n"
                    f"Answer: {answer}"
                ),
            },
        ]
        result = await self._call_judge(
            messages,
            {"type": "array", "items": {"type": "string"}},
            "answer-relevancy",
        )
        tokens = result.usage.get("total_tokens", 0) if result.usage else 0
        if not result.structured or not result.structured.success:
            logger.error(
                "❌ Reverse question gen failed: %s | raw=%r",
                result.structured.error
                if result.structured
                else "No structured output",
                result.content[:200],
            )
            return [], tokens
        logger.debug("❓ Generated %d reverse questions", len(result.structured.parsed))
        return result.structured.parsed, tokens
