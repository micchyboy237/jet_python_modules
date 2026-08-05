# jet_python_modules/jet/adapters/llama_cpp/budget_utils.py
"""Pre-flight token budget validation for llama.cpp RAG pipelines.

Prevents HTTP 400 errors from llama.cpp server by validating total prompt
tokens against model context window BEFORE sending requests. Accounts for
system prompt, user query, retrieved chunks, chat template overhead, and
reserved completion tokens.

Uses existing primitives:
- token_utils.count_chat_tokens: Accurate measurement including special tokens
- models.LLAMACPP_MODEL_CONTEXTS: Static context lookup (no network)
- model_utils.get_model_ctx_embd_size: Server fallback for unknown models
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from jet.adapters.llama_cpp.model_utils import get_model_ctx_embd_size
from jet.adapters.llama_cpp.models import LLAMACPP_MODEL_CONTEXTS
from jet.adapters.llama_cpp.token_utils import count_chat_tokens, count_tokens
from jet.adapters.llama_cpp.types import LLAMACPP_KEYS

logger = logging.getLogger(__name__)

_DEFAULT_MAX_COMPLETION_TOKENS = 512
_DEFAULT_SYSTEM_PROMPT_TOKENS = 64


@dataclass
class BudgetAllocation:
    """Breakdown of token budget allocation for traceability."""

    model_ctx: int
    system_tokens: int
    query_tokens: int
    chunk_tokens: int
    completion_reserve: int
    chat_template_overhead: int
    total_used: int
    available_for_chunks: int
    chunks_included: int
    chunks_truncated: int
    within_budget: bool


@dataclass
class PromptBudget:
    """Pre-flight budget validator for llama.cpp RAG pipelines.

    Validates that system + query + chunks + completion fit within the
    model's context window. Truncates lowest-priority chunks (end of list)
    when budget is exceeded.

    Usage:
        budget = PromptBudget("qwen3.5:2b", max_completion_tokens=512)
        safe_chunks = budget.validate(system_prompt, query, chunks)
        # safe_chunks is guaranteed to fit within budget
    """

    model: str | LLAMACPP_KEYS
    max_completion_tokens: int = _DEFAULT_MAX_COMPLETION_TOKENS
    safety_margin: int = 16

    def __post_init__(self) -> None:
        self._ctx = self._resolve_context(self.model)
        logger.info(
            "PromptBudget initialized: model=%s, ctx=%d, completion_reserve=%d, margin=%d",
            self.model,
            self._ctx,
            self.max_completion_tokens,
            self.safety_margin,
        )

    @property
    def context_window(self) -> int:
        """Total context window size in tokens."""
        return self._ctx

    def available_for_chunks(
        self,
        system_prompt: str = "",
        query: str = "",
    ) -> int:
        """Calculate remaining token budget for retrieved chunks.

        Args:
            system_prompt: System prompt text.
            query: User query text.

        Returns:
            Tokens available for chunks after reserving system, query,
            completion, and safety margin. Always >= 0.
        """
        system_tok = (
            count_tokens(system_prompt, add_special=False, model=self.model)
            if system_prompt
            else 0
        )
        query_tok = (
            count_tokens(query, add_special=False, model=self.model) if query else 0
        )

        available = (
            self._ctx
            - system_tok
            - query_tok
            - self.max_completion_tokens
            - self.safety_margin
        )
        return max(0, available)

    def validate(
        self,
        system_prompt: str,
        query: str,
        chunks: List[str],
        max_chunks: Optional[int] = None,
    ) -> List[str]:
        """Validate and truncate chunks to fit within token budget.
        Measures actual token counts including chat template overhead.
        Removes chunks from the END of the list (lowest priority) until
        the total fits within budget.

        Args:
            system_prompt: System prompt text.
            query: User query text.
            chunks: Retrieved chunks ordered by relevance (best first).
            max_chunks: Hard cap on number of chunks regardless of budget.
                       Defaults to 5 if not specified.

        Returns:
            Subset of chunks guaranteed to fit within budget.
            May be empty if even a single chunk exceeds available budget.
        """
        if max_chunks is None:
            max_chunks = 5

        capped_chunks = chunks[:max_chunks]

        if not capped_chunks:
            logger.debug("validate: no chunks provided, returning empty list")
            return []

        # Measure fixed costs
        system_tok = (
            count_tokens(system_prompt, add_special=False, model=self.model)
            if system_prompt
            else 0
        )
        query_tok = (
            count_tokens(query, add_special=False, model=self.model) if query else 0
        )

        # Build messages to measure chat template overhead accurately
        messages = (
            [{"role": "system", "content": system_prompt}] if system_prompt else []
        )
        messages.append({"role": "user", "content": query})

        try:
            chat_result = count_chat_tokens(messages, model=self.model)
            base_msg_tokens = (
                chat_result["input_tokens"]
                if isinstance(chat_result, dict) and "input_tokens" in chat_result
                else int(chat_result)
            )
        except Exception as exc:
            logger.warning(
                "count_chat_tokens failed (%s), falling back to raw count", exc
            )
            base_msg_tokens = system_tok + query_tok

        chat_overhead = max(0, base_msg_tokens - system_tok - query_tok)

        fixed_cost = base_msg_tokens + self.max_completion_tokens + self.safety_margin
        chunk_budget = max(0, self._ctx - fixed_cost)

        logger.debug(
            "Budget breakdown: ctx=%d, system=%d, query=%d, overhead=%d, "
            "completion=%d, margin=%d → chunk_budget=%d",
            self._ctx,
            system_tok,
            query_tok,
            chat_overhead,
            self.max_completion_tokens,
            self.safety_margin,
            chunk_budget,
        )

        # Greedily include chunks from front (highest relevance) until budget exhausted
        included: List[str] = []
        running_chunk_tokens = 0

        for i, chunk in enumerate(capped_chunks):
            chunk_tok = count_tokens(chunk, add_special=False, model=self.model)

            # Account for message framing overhead per chunk (~4 tokens for role/content delimiters)
            chunk_msg_overhead = 4
            total_chunk_cost = chunk_tok + chunk_msg_overhead

            if running_chunk_tokens + total_chunk_cost > chunk_budget:
                logger.info(
                    "Budget truncation: stopping at chunk %d/%d "
                    "(used=%d + next=%d > budget=%d)",
                    i,
                    len(capped_chunks),
                    running_chunk_tokens,
                    total_chunk_cost,
                    chunk_budget,
                )
                break

            included.append(chunk)
            running_chunk_tokens += total_chunk_cost

        allocation = BudgetAllocation(
            model_ctx=self._ctx,
            system_tokens=system_tok,
            query_tokens=query_tok,
            chunk_tokens=running_chunk_tokens,
            completion_reserve=self.max_completion_tokens,
            chat_template_overhead=chat_overhead,
            total_used=base_msg_tokens
            + running_chunk_tokens
            + self.max_completion_tokens,
            available_for_chunks=chunk_budget,
            chunks_included=len(included),
            chunks_truncated=len(capped_chunks) - len(included),
            within_budget=(
                base_msg_tokens + running_chunk_tokens + self.max_completion_tokens
                <= self._ctx
            ),
        )

        logger.info(
            "Budget validation: %d/%d chunks included (%d tokens used / %d ctx), "
            "truncated=%d, within_budget=%s",
            allocation.chunks_included,
            len(capped_chunks),
            allocation.total_used,
            self._ctx,
            allocation.chunks_truncated,
            allocation.within_budget,
        )

        if not allocation.within_budget:
            logger.error(
                "CRITICAL: Even after truncation, prompt may exceed context! "
                "total_used=%d > ctx=%d. Reduce max_completion_tokens or chunk count.",
                allocation.total_used,
                self._ctx,
            )

        return included

    def get_allocation(
        self,
        system_prompt: str,
        query: str,
        chunks: List[str],
        max_chunks: Optional[int] = None,
    ) -> BudgetAllocation:
        """Get detailed budget allocation without modifying chunks.

        Useful for logging/debugging before calling validate().
        """
        validated = self.validate(system_prompt, query, chunks, max_chunks)

        system_tok = (
            count_tokens(system_prompt, add_special=False, model=self.model)
            if system_prompt
            else 0
        )
        query_tok = (
            count_tokens(query, add_special=False, model=self.model) if query else 0
        )
        chunk_tok = sum(
            count_tokens(c, add_special=False, model=self.model) for c in validated
        )

        messages = (
            [{"role": "system", "content": system_prompt}] if system_prompt else []
        )
        messages.append({"role": "user", "content": query})
        try:
            chat_result = count_chat_tokens(messages, model=self.model)
            base_msg_tokens = (
                chat_result["input_tokens"]
                if isinstance(chat_result, dict) and "input_tokens" in chat_result
                else int(chat_result)
            )
        except Exception:
            base_msg_tokens = system_tok + query_tok

        chat_overhead = max(0, base_msg_tokens - system_tok - query_tok)

        return BudgetAllocation(
            model_ctx=self._ctx,
            system_tokens=system_tok,
            query_tokens=query_tok,
            chunk_tokens=chunk_tok,
            completion_reserve=self.max_completion_tokens,
            chat_template_overhead=chat_overhead,
            total_used=base_msg_tokens + chunk_tok + self.max_completion_tokens,
            available_for_chunks=max(
                0,
                self._ctx
                - base_msg_tokens
                - self.max_completion_tokens
                - self.safety_margin,
            ),
            chunks_included=len(validated),
            chunks_truncated=len(chunks[: max_chunks or 5]) - len(validated),
            within_budget=(
                base_msg_tokens + chunk_tok + self.max_completion_tokens <= self._ctx
            ),
        )

    @staticmethod
    def _resolve_context(model: str | LLAMACPP_KEYS) -> int:
        """Resolve model context size using static dict first, server fallback."""
        if model in LLAMACPP_MODEL_CONTEXTS:
            ctx = LLAMACPP_MODEL_CONTEXTS[model]
            logger.debug("Static context for %s: %d", model, ctx)
            return ctx

        try:
            info = get_model_ctx_embd_size(model)
            ctx = info.get("ctx", 0)
            if ctx > 0:
                logger.debug("Server context for %s: %d", model, ctx)
                return ctx
        except Exception as exc:
            logger.warning(
                "Failed to resolve context for %s (%s: %s)",
                model,
                type(exc).__name__,
                exc,
            )

        logger.warning("Using fallback context 4096 for unknown model %s", model)
        return 4096
