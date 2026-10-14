"""Browser-Use LLM adapter for local llama.cpp servers using jet observability."""

from __future__ import annotations

import re
from typing import Any, TypeVar

from browser_use.llm.base import BaseChatModel
from browser_use.llm.messages import AssistantMessage, BaseMessage, SystemMessage
from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage
from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.llama_cpp.llm_utils_observed import achat
from jet.libs.llama_cpp.usage.chat_stream_types import StreamCompletionResult
from jet.libs.llama_cpp.usage.structured_output import (
    parse_structured_content,
    resolve_response_format,
)
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class ChatLlamaCpp(BaseChatModel):
    """A Browser-Use compatible chat model backed by jet's observed llama.cpp utilities."""

    def __init__(
        self,
        model: str = LLM_MODEL,
        base_url: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 8192,
        top_p: float = 0.8,
        top_k: int = 20,
        repeat_penalty: float = 1.1,
        enable_thinking: bool = False,
        **kwargs: Any,
    ):
        self.model = model
        self._base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.repeat_penalty = repeat_penalty
        self.enable_thinking = enable_thinking
        self.extra_kwargs = kwargs

    @property
    def provider(self) -> str:
        return "llama_cpp"

    @property
    def name(self) -> str:
        return f"llama_cpp/{self.model}"

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        output_format: type[T] | None = None,
        **kwargs: Any,
    ) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
        """Invoke the local llama.cpp model with Browser-Use messages."""
        # 1. Convert messages
        openai_messages = []
        for msg in messages:
            role = "user"
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, AssistantMessage):
                role = "assistant"

            content = msg.content
            if isinstance(content, list):
                text_parts = [
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                content = "\n".join(text_parts)

            openai_messages.append({"role": role, "content": content})

        # 2. Resolve format
        resolved_format = (
            resolve_response_format(output_format) if output_format else None
        )

        # 3. Call jet's observed achat
        result: StreamCompletionResult = await achat(
            prompt_or_messages=openai_messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            repeat_penalty=self.repeat_penalty,
            enable_thinking=self.enable_thinking,
            response_format=output_format,
            project_name="browser-use-agent",
            **self.extra_kwargs,
        )

        # 4. Map Usage (FIX: Provide defaults for missing fields to prevent Pydantic errors)
        usage = None
        if result.usage:
            usage = ChatInvokeUsage(
                prompt_tokens=result.usage.get("prompt_tokens", 0),
                completion_tokens=result.usage.get("completion_tokens", 0),
                total_tokens=result.usage.get("total_tokens", 0),
                # Defaulting to 0 prevents the ValidationError seen in traces
                prompt_cached_tokens=result.usage.get("prompt_cached_tokens", 0)
                or result.usage.get("cached_tokens", 0),
                prompt_cache_creation_tokens=result.usage.get(
                    "prompt_cache_creation_tokens", 0
                ),
                prompt_image_tokens=result.usage.get("prompt_image_tokens", 0),
            )

        # 5. Handle Structured Output
        if output_format is not None:
            # Try jet's built-in structured result first
            if result.structured and result.structured.success:
                return ChatInvokeCompletion(
                    completion=result.structured.parsed,
                    usage=usage,
                    stop_reason=result.finish_reason,
                )

            # Fallback: Manual extraction if jet failed but we have content
            if result.content:
                # Clean up common tag leakage from small models before parsing
                cleaned_content = re.sub(r"<[^>]+>", "", result.content)
                parsed_result = parse_structured_content(
                    cleaned_content, resolved_format
                )

                if parsed_result.success:
                    return ChatInvokeCompletion(
                        completion=parsed_result.parsed,
                        usage=usage,
                        stop_reason=result.finish_reason,
                    )

            # Raise error if all attempts fail
            raise ValueError(
                f"Failed to parse structured output: {result.structured.error if result.structured else 'No content'}"
            )

        return ChatInvokeCompletion(
            completion=result.content,
            usage=usage,
            stop_reason=result.finish_reason,
        )
