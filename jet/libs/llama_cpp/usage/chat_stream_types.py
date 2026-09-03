from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jet.libs.llama_cpp.usage.structured_output import StructuredResult


@dataclass
class ToolCallResult:
    """A fully accumulated tool call from streaming deltas."""

    id: str
    type: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str


@dataclass
class StreamCompletionResult:
    """Structured result from a streamed chat completion."""

    content: str
    tool_calls: list[ToolCallResult] = field(default_factory=list)
    usage: dict[str, int] | None = None
    finish_reason: str | None = None
    structured: StructuredResult | None = None  # NEW: Explicit field

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0
