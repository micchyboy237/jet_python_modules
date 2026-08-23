"""Accumulation memory for ReAct Web Searcher with token-accurate budgeting.

Maintains structured state across ReAct loop iterations to enable:
- Intermediate context sufficiency evaluation
- List-intent item tracking
- Duplicate search/URL prevention
- Budget-aware context accumulation using actual token counts

Reuses existing infrastructure:
- count_tokens(): Accurate local tokenization (no server round-trip)
- get_model_ctx_embd_size(): Dynamic context window derivation
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from jet.adapters.llama_cpp.model_utils import get_model_ctx_embd_size
from jet.adapters.llama_cpp.token_utils import count_tokens

from .types import QueryIntent, SearchResult

logger = logging.getLogger(__name__)

# Regex patterns for extracting list items from text
_LIST_ITEM_PATTERNS = [
    re.compile(r"^\s*\d+[\.\)]\s+(.+)$", re.MULTILINE),
    re.compile(r"^\s*[-•]\s+(.+)$", re.MULTILINE),
]

# Fraction of model context reserved for accumulated memory
# Leaves 75% for system prompt + tools + generation
_MEMORY_CTX_FRACTION = 0.25


@dataclass(frozen=True)
class SufficiencySnapshot:
    """Immutable snapshot of the last sufficiency evaluation."""

    is_sufficient: bool
    missing_info: tuple[str, ...]
    next_action: str
    suggested_query: str
    contexts_evaluated: int
    tokens_evaluated: int
    timestamp_ns: int = 0


class AccumulationMemory:
    """Structured memory with token-accurate budget tracking.

    Uses count_tokens() for precise budget enforcement and
    get_model_ctx_embd_size() to derive limits from model capabilities.
    """

    def __init__(
        self,
        model: str,
        intent: QueryIntent = QueryIntent.UNKNOWN,
        target_list_size: int | None = None,
        max_accumulated_tokens: int | None = None,
    ) -> None:
        self._model = model
        self._intent = intent
        self._target_list_size = target_list_size

        # Derive token budget from model context window
        if max_accumulated_tokens is not None:
            self._max_tokens = max_accumulated_tokens
        else:
            try:
                ctx_info = get_model_ctx_embd_size(model)
                ctx = ctx_info.get("ctx", 0)
                if ctx > 0:
                    self._max_tokens = max(int(ctx * _MEMORY_CTX_FRACTION), 512)
                else:
                    self._max_tokens = 2048
            except Exception as exc:
                logger.warning(
                    "⚠️ Could not derive memory budget for %s (%s); using default 2048",
                    model,
                    exc,
                )
                self._max_tokens = 2048

        # Core accumulation state
        self._contexts: list[str] = []
        self._context_tokens: int = 0
        self._visited_urls: set[str] = set()
        self._search_queries: list[str] = []

        # List-intent specific state
        self._list_items: list[str] = []

        # Sufficiency evaluation cache
        self._last_sufficiency: SufficiencySnapshot | None = None

        logger.info(
            "🧠 AccumulationMemory initialized: model=%s, intent=%s, "
            "target_list=%s, max_tokens=%d",
            model,
            intent.value,
            target_list_size,
            self._max_tokens,
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def intent(self) -> QueryIntent:
        return self._intent

    @property
    def num_contexts(self) -> int:
        return len(self._contexts)

    @property
    def accumulated_tokens(self) -> int:
        return self._context_tokens

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def num_visited_urls(self) -> int:
        return len(self._visited_urls)

    @property
    def num_searches(self) -> int:
        return len(self._search_queries)

    @property
    def list_items(self) -> list[str]:
        return list(self._list_items)

    @property
    def list_item_count(self) -> int:
        return len(self._list_items)

    @property
    def is_budget_exhausted(self) -> bool:
        return self._context_tokens >= self._max_tokens

    @property
    def remaining_token_budget(self) -> int:
        return max(0, self._max_tokens - self._context_tokens)

    # ── Recording Methods ──────────────────────────────────────────────────

    def record_search(
        self,
        query: str,
        results: list[SearchResult] | None = None,
        observation: str = "",
    ) -> None:
        """Record a search execution and its results."""
        self._search_queries.append(query)

        if observation:
            self._add_context(observation)

        if self._intent == QueryIntent.LIST and results:
            for r in results:
                self._try_extract_list_item(r.title)
                if r.snippet:
                    self._try_extract_list_item(r.snippet)

        logger.debug(
            "📝 Recorded search: query=%r, results=%d, total_searches=%d, tokens=%d/%d",
            query[:60],
            len(results) if results else 0,
            len(self._search_queries),
            self._context_tokens,
            self._max_tokens,
        )

    def record_read(
        self,
        url: str,
        content: str,
        tokens: int = 0,
        title: str | None = None,
    ) -> None:
        """Record a URL read with optional pre-computed token count."""
        self._visited_urls.add(url)

        if content:
            if tokens <= 0:
                tokens = count_tokens(content, model=self._model)
            self._add_context_with_tokens(content, tokens)

        if self._intent == QueryIntent.LIST:
            if title:
                self._try_extract_list_item(title)
            if content:
                self._extract_list_items_from_text(content)

        logger.debug(
            "📝 Recorded read: url=%s, tokens=%d, total_urls=%d, "
            "list_items=%d, budget=%d/%d",
            url[:60],
            tokens,
            len(self._visited_urls),
            len(self._list_items),
            self._context_tokens,
            self._max_tokens,
        )

    def record_synthesis(self, answer: str) -> None:
        """Record that synthesis was attempted. Resets sufficiency cache."""
        self._last_sufficiency = None
        logger.debug("📝 Recorded synthesis attempt (%d chars)", len(answer))

    # ── Query Methods ──────────────────────────────────────────────────────

    def has_visited(self, url: str) -> bool:
        return url in self._visited_urls

    def has_searched(self, query: str) -> bool:
        return query in self._search_queries

    def is_duplicate_search(self, query: str) -> bool:
        """Detect near-duplicate queries in recent history."""
        q_lower = query.lower().strip()
        for prev in self._search_queries[-5:]:
            prev_lower = prev.lower().strip()
            if q_lower == prev_lower:
                return True
            if len(q_lower) > 10 and len(prev_lower) > 10:
                if q_lower in prev_lower or prev_lower in q_lower:
                    return True
        return False

    def is_per_entity_loop(self, threshold: int = 3) -> bool:
        """Detect if recent searches target individual entities in a list query."""
        if len(self._search_queries) < threshold:
            return False

        recent = self._search_queries[-threshold:]
        entity_pattern = re.compile(r"(season|part|vol\.?|volume)\s*\d+", re.IGNORECASE)
        entity_count = sum(1 for q in recent if entity_pattern.search(q))
        return entity_count >= threshold

    def is_zero_result_loop(self, threshold: int = 3) -> bool:
        """Detect if recent searches all returned zero results."""
        if len(self._contexts) < threshold:
            return False
        recent_obs = self._contexts[-threshold:]
        zero_count = sum(
            1
            for obs in recent_obs
            if "No search results found" in obs or "no results" in obs.lower()
        )
        return zero_count >= threshold

    def get_accumulated_contexts(self) -> list[str]:
        """Return all accumulated context texts for sufficiency evaluation."""
        return list(self._contexts)

    def get_sufficiency_snapshot(self) -> SufficiencySnapshot | None:
        return self._last_sufficiency

    def update_sufficiency(
        self,
        is_sufficient: bool,
        missing_info: list[str],
        next_action: str,
        suggested_query: str = "",
    ) -> SufficiencySnapshot:
        """Cache a new sufficiency evaluation result."""
        snapshot = SufficiencySnapshot(
            is_sufficient=is_sufficient,
            missing_info=tuple(missing_info),
            next_action=next_action,
            suggested_query=suggested_query,
            contexts_evaluated=len(self._contexts),
            tokens_evaluated=self._context_tokens,
            timestamp_ns=time.time_ns(),
        )
        self._last_sufficiency = snapshot

        logger.debug(
            "💾 Sufficiency cached: sufficient=%s, action=%s, gaps=%d, tokens=%d",
            is_sufficient,
            next_action,
            len(missing_info),
            self._context_tokens,
        )
        return snapshot

    def should_force_synthesis(self) -> bool:
        """Determine if the agent should be forced to synthesize now."""
        if not self._contexts:
            return False

        # Zero-result loop detection
        if self.is_zero_result_loop() and self._context_tokens > 128:
            logger.info(
                "🛑 Force synthesis: zero-result loop detected (%d consecutive failures)",
                len(self._search_queries),
            )
            return True

        # Budget exhausted with meaningful content
        if self.is_budget_exhausted and self._context_tokens > 256:
            logger.info(
                "🛑 Force synthesis: budget exhausted (%d/%d tokens)",
                self._context_tokens,
                self._max_tokens,
            )
            return True

        # List intent target reached
        if (
            self._intent == QueryIntent.LIST
            and self._target_list_size is not None
            and len(self._list_items) >= self._target_list_size
        ):
            logger.info(
                "🛑 Force synthesis: list target reached (%d/%d items)",
                len(self._list_items),
                self._target_list_size,
            )
            return True

        # Per-entity loop with existing context
        if self.is_per_entity_loop() and self._context_tokens > 256:
            logger.info(
                "🛑 Force synthesis: per-entity loop detected with %d tokens",
                self._context_tokens,
            )
            return True

        return False

    def get_status_summary(self) -> dict[str, Any]:
        return {
            "model": self._model,
            "intent": self._intent.value,
            "contexts": len(self._contexts),
            "tokens_used": self._context_tokens,
            "tokens_max": self._max_tokens,
            "tokens_remaining": self.remaining_token_budget,
            "urls_visited": len(self._visited_urls),
            "searches": len(self._search_queries),
            "list_items": len(self._list_items),
            "budget_exhausted": self.is_budget_exhausted,
            "has_sufficiency": self._last_sufficiency is not None,
        }

    # ── Internal Helpers ───────────────────────────────────────────────────

    def _add_context(self, text: str) -> None:
        """Add context with automatic token counting."""
        if not text or not text.strip():
            return
        stripped = text.strip()
        tokens = count_tokens(stripped, model=self._model)
        self._add_context_with_tokens(stripped, tokens)

    def _add_context_with_tokens(self, text: str, tokens: int) -> None:
        """Add context with pre-computed token count and budget enforcement."""
        if not text or not text.strip():
            return

        stripped = text.strip()

        # Deduplicate
        if stripped in self._contexts:
            logger.debug("⏭️ Skipping duplicate context (%d tokens)", tokens)
            return

        # Budget check
        if self._context_tokens + tokens > self._max_tokens:
            remaining = self._max_tokens - self._context_tokens
            if remaining < 64:
                logger.debug(
                    "⏭️ Token budget full (%d/%d), skipping %d token addition",
                    self._context_tokens,
                    self._max_tokens,
                    tokens,
                )
                return

            # Truncate to fit remaining budget
            char_ratio = len(stripped) / max(tokens, 1)
            truncated_chars = int(remaining * char_ratio)
            stripped = stripped[:truncated_chars]
            tokens = count_tokens(stripped, model=self._model)

            if tokens > remaining:
                logger.debug(
                    "⏭️ Truncated context still exceeds budget (%d > %d remaining)",
                    tokens,
                    remaining,
                )
                return

            logger.debug(
                "✂️ Truncated context to fit budget: %d → %d tokens",
                self._context_tokens + tokens,
                self._context_tokens + tokens,
            )

        self._contexts.append(stripped)
        self._context_tokens += tokens

        logger.debug(
            "➕ Added context: %d tokens (total: %d/%d tokens, %d contexts)",
            tokens,
            self._context_tokens,
            self._max_tokens,
            len(self._contexts),
        )

    def _try_extract_list_item(self, text: str) -> None:
        if not text or not text.strip():
            return
        cleaned = text.strip()
        if len(cleaned) < 3 or len(cleaned) > 200:
            return

        normalized = cleaned.lower()
        existing_normalized = {item.lower() for item in self._list_items}

        if normalized not in existing_normalized:
            self._list_items.append(cleaned)
            logger.debug(
                "📋 Extracted list item #%d: %r",
                len(self._list_items),
                cleaned[:60],
            )

    def _extract_list_items_from_text(self, text: str) -> None:
        for pattern in _LIST_ITEM_PATTERNS:
            matches = pattern.findall(text)
            for match in matches:
                self._try_extract_list_item(match)
