# jet_python_modules/jet/libs/llama_cpp/usage/context_window.py
"""Encapsulated context window management for stateful agents.

Provides a reusable, testable ContextWindow class that handles:
- Message history storage and retrieval
- Accurate token counting via jet.adapters.llama_cpp.token_utils (local HF tokenizer)
- Safety-first truncation (preserves system prompts)
- Multimodal content injection
- Selective history clearing
"""

from __future__ import annotations

import logging
from typing import Any

from jet.adapters.llama_cpp.token_utils import count_tokens
from jet.libs.llama_cpp.usage.chat_stream_observability import MODEL as DEFAULT_MODEL

logger = logging.getLogger(__name__)

# Conservative vision token estimate when images can't be precisely counted
IMAGE_TOKEN_ESTIMATE = 256


class ContextWindow:
    """Manages conversation history within a token budget.

    Uses accurate local tokenization from jet.adapters.llama_cpp.token_utils,
    supporting chat templates, tool definitions, and cached HF tokenizers.

    Attributes:
        max_tokens: Maximum allowed tokens in the context window.
        model: Model name used for tokenization.
        base_url: Optional server base URL for remote token counting.
        history: Ordered list of message dicts (OpenAI format).
    """

    def __init__(
        self,
        max_tokens: int = 16384,
        model: str = DEFAULT_MODEL,
        base_url: str | None = None,
    ):
        self.max_tokens = max_tokens
        self.model = model
        self.base_url = base_url
        self.history: list[dict[str, Any]] = []
        logger.debug(
            f"📦 ContextWindow initialized (max_tokens={max_tokens}, model={model}, base_url={base_url})"
        )

    # ── Token Counting ────────────────────────────────────────────────

    def _count_message_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Count tokens for a list of messages using token_utils.
        Falls back to per-message estimation if batch counting fails
        (e.g., due to multimodal content that chat templates can't handle).
        """
        try:
            return count_tokens(
                messages, model=self.model, base_url=self.base_url, use_server=True
            )
        except Exception as e:
            logger.debug(
                f"Batch token count failed ({e}), falling back to per-message estimation"
            )
            return self._estimate_message_tokens(messages)

    @staticmethod
    def _estimate_message_tokens(messages: list[dict[str, Any]]) -> int:
        """Fallback token estimation for messages that can't be template-counted.

        Handles multimodal content blocks gracefully.
        """
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(content) // 4
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        total += len(str(block.get("text", ""))) // 4
                    elif block.get("type") == "image_url":
                        total += IMAGE_TOKEN_ESTIMATE
            else:
                total += len(str(content)) // 4
        return total

    def total_tokens(self) -> int:
        """Sum estimated tokens across entire history."""
        if not self.history:
            return 0
        return self._count_message_tokens(self.history)

    # ── History Mutation ──────────────────────────────────────────────

    def append(self, message: dict[str, Any]) -> None:
        """Append a single message to history."""
        self.history.append(message)
        role = message.get("role", "?")
        tokens = self._count_message_tokens([message])
        logger.debug(f"➕ Appended {role} message (~{tokens} tokens)")

    def append_image(
        self, prompt: str | None, base64_data: str, mime_type: str
    ) -> None:
        """Append a multimodal user message with embedded image."""
        content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt or ""},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{base64_data}"},
            },
        ]
        self.append({"role": "user", "content": content})
        logger.info(
            f"🖼️ Added image to context ({mime_type}, ~{IMAGE_TOKEN_ESTIMATE} tokens)"
        )

    def clear(self, preserve_system: bool = True) -> None:
        """Reset history, optionally keeping system messages."""
        if preserve_system:
            self.history = [m for m in self.history if m.get("role") == "system"]
            logger.info(f"🧹 Context cleared (kept {len(self.history)} system msg(s))")
        else:
            self.history = []
            logger.info("🧹 Context fully cleared")

    # ── Truncation ────────────────────────────────────────────────────

    def truncate_if_needed(self) -> bool:
        """Truncate oldest non-system messages if over token budget.

        Returns:
            True if truncation occurred, False otherwise.
        """
        current = self.total_tokens()
        if current <= self.max_tokens:
            return False

        system_msgs = [m for m in self.history if m.get("role") == "system"]
        non_system = [m for m in self.history if m.get("role") != "system"]

        kept: list[dict[str, Any]] = []
        running_total = self._count_message_tokens(system_msgs) if system_msgs else 0

        # Walk backwards: keep newest messages that fit
        for msg in reversed(non_system):
            msg_tokens = self._count_message_tokens([msg])
            if running_total + msg_tokens > self.max_tokens:
                continue
            kept.append(msg)
            running_total += msg_tokens

        original_count = len(self.history)
        self.history = system_msgs + list(reversed(kept))
        new_count = len(self.history)
        logger.info(
            f"🧹 Truncated context: {current}→{running_total} tokens, "
            f"{original_count}→{new_count} messages"
        )
        return True

    # ── Read-only Helpers ─────────────────────────────────────────────

    @property
    def message_count(self) -> int:
        return len(self.history)

    @property
    def system_message_count(self) -> int:
        return sum(1 for m in self.history if m.get("role") == "system")

    def get_messages(self) -> list[dict[str, Any]]:
        """Return a shallow copy of current history."""
        return list(self.history)
