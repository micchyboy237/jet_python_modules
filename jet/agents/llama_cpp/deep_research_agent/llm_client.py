"""Centralized LLM client for Deep Research Agent with safety guards.

Reuses robust patterns from llm_as_a_judge:
- Dynamic context budget calculation via get_model_ctx_embd_size
- Sentence-aware truncation via truncate_texts
- Truncation detection via finish_reason
- Standardized structured output error handling
"""

from __future__ import annotations

import logging
from typing import Any

from jet.adapters.llama_cpp.chunking_utils import truncate_texts
from jet.adapters.llama_cpp.llm_utils import achat
from jet.adapters.llama_cpp.model_utils import get_model_ctx_embd_size
from jet.libs.llama_cpp.usage.chat_stream_types import StreamCompletionResult

logger = logging.getLogger(__name__)

# Reserve tokens for generation/output parsing overhead
RESERVE_TOKENS = 1024


async def safe_achat(
    messages: list[dict[str, Any]],
    model: str,
    response_format: Any,
    *,
    temperature: float = 0.1,
    max_tokens: int | None = None,
    context_text: str | None = None,
    metric_name: str = "agent-call",
) -> StreamCompletionResult:
    """Wrapper around achat with truncation detection and context safety.

    Args:
        messages: OpenAI-format messages list.
        model: Model identifier for llama.cpp server.
        response_format: Pydantic model, JSON Schema dict, or None.
        temperature: Sampling temperature (default 0.1 for structured output).
        max_tokens: Max generation tokens. Defaults to 1024 if None.
        context_text: If provided, will be truncated to fit model budget
                      before being included in messages. Matches against
                      message content to perform in-place replacement.
        metric_name: Label for logging/tracing.

    Returns:
        StreamCompletionResult with structured output parsed if successful.
    """
    effective_max_tokens = max_tokens or 1024

    # --- Dynamic Budget Calculation ---
    try:
        ctx_info = get_model_ctx_embd_size(model)
        budget = max(ctx_info["ctx"] - RESERVE_TOKENS, 512)
    except Exception:
        logger.warning("Could not get ctx size for %s, using default 3072", model)
        budget = 3072

    # --- Safe Context Truncation ---
    if context_text and len(context_text) > budget * 4:  # rough char→token estimate
        logger.debug(
            "[%s] Truncating context (%d chars) to budget %d tokens",
            metric_name,
            len(context_text),
            budget,
        )
        truncated = truncate_texts(
            context_text,
            model=model,
            max_tokens=budget,
            strict_sentences=True,
            show_progress=False,
        )
        # truncate_texts may return a list; normalize to single string
        if isinstance(truncated, list):
            truncated = truncated[0] if truncated else ""

        # Replace original context in messages with truncated version
        for msg in reversed(messages):
            if msg["role"] == "user" and context_text in msg.get("content", ""):
                msg["content"] = msg["content"].replace(context_text, truncated)
                logger.debug(
                    "[%s] Context replaced in user message (%d → %d chars)",
                    metric_name,
                    len(context_text),
                    len(truncated),
                )
                break

    # --- LLM Call ---
    result = await achat(
        prompt_or_messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=effective_max_tokens,
        response_format=response_format,
        enable_thinking=False,
        seed=42,
    )

    tokens = result.usage.get("total_tokens", 0) if result.usage else 0

    # --- Truncation Detection ---
    if result.finish_reason == "length":
        logger.warning(
            "⚠️ [%s] Truncated at max_tokens=%d (tokens=%d)",
            metric_name,
            effective_max_tokens,
            tokens,
        )

    # --- Structured Output Validation Logging ---
    if not result.structured or not result.structured.success:
        error = result.structured.error if result.structured else "No structured output"
        logger.error(
            "❌ [%s] Parse failed: %s | raw=%r",
            metric_name,
            error,
            result.content[:200],
        )
    else:
        logger.debug(
            "✅ [%s] Structured parse OK (tokens=%d)",
            metric_name,
            tokens,
        )

    return result
