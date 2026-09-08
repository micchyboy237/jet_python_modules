"""Chonkie-compatible Genie adapter for llama.cpp servers.

Bridges jet.adapters.llama_cpp.llm_utils with chonkie.genie.BaseGenie,
enabling text generation and structured JSON output via local/remote
llama.cpp OpenAI-compatible servers.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any, Optional

from jet.adapters.llama_cpp.config import LLM_BASE_URL, LLM_MODEL
from jet.logger import logger

from chonkie.genie.base import BaseGenie

if TYPE_CHECKING:
    from pydantic import BaseModel


class LlamacppGenie(BaseGenie):
    """Genie backed by a llama.cpp OpenAI-compatible server.

    Delegates to jet.adapters.llama_cpp.llm_utils for streaming chat
    completions, structured output validation, tool-use loops, and
    vision input support.

    Args:
        model: Model identifier served by llama.cpp. Defaults to
            LLAMA_CPP_LLM_MODEL env var or "qwen3.5-uncensored:2b".
        base_url: Override the LLM server URL. If None, uses
            LLAMA_CPP_LLM_URL / LLAMA_CPP_LLM_HOST env vars.
        temperature: Sampling temperature. Default 0.7. Use 0.0–0.3
            for reliable structured output.
        max_tokens: Maximum tokens to generate. Default 16384.
        enable_thinking: Request reasoning tokens. Must be False when
            using grammar-based structured output.
        system_prompt: Optional system message prepended to every request.
    """

    def __init__(
        self,
        model: str = LLM_MODEL,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 16384,
        enable_thinking: bool = False,
        system_prompt: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.base_url = base_url or LLM_BASE_URL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.system_prompt = system_prompt

        # Pre-create clients to avoid per-call overhead
        from jet.adapters.llama_cpp.factory import (
            get_async_llm_client,
            get_llm_client,
        )

        self._client = get_llm_client(base_url=self.base_url)
        self._async_client = get_async_llm_client(base_url=self.base_url)

        logger.info(
            f"LlamacppGenie initialized: model={model}, base_url={self.base_url}"
        )

    def _build_messages(self, prompt: str) -> list[dict[str, str]]:
        """Construct messages list with optional system prompt."""
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(self, prompt: str) -> str:
        """Generate a plain-text response via synchronous streaming chat."""
        from jet.adapters.llama_cpp.llm_utils import chat

        result = chat(
            prompt_or_messages=self._build_messages(prompt),
            model=self.model,
            client=self._client,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            enable_thinking=self.enable_thinking,
        )
        if not result.content:
            raise ValueError(
                f"LlamacppGenie received empty response "
                f"(finish_reason={result.finish_reason})"
            )
        return result.content

    async def agenerate(self, prompt: str) -> str:
        """Generate a plain-text response via asynchronous streaming chat."""
        from jet.adapters.llama_cpp.llm_utils import achat

        result = await achat(
            prompt_or_messages=self._build_messages(prompt),
            model=self.model,
            client=self._async_client,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            enable_thinking=self.enable_thinking,
        )
        if not result.content:
            raise ValueError(
                f"LlamacppGenie received empty async response "
                f"(finish_reason={result.finish_reason})"
            )
        return result.content

    def generate_json(self, prompt: str, schema: "BaseModel") -> dict[str, Any]:
        """Generate a structured JSON response validated against a Pydantic schema.

        Args:
            prompt: User prompt describing the desired output.
            schema: Pydantic BaseModel class defining the expected structure.

        Returns:
            dict matching the schema fields.

        Raises:
            ValueError: If structured output parsing/validation fails.
        """
        from jet.adapters.llama_cpp.llm_utils import chat

        result = chat(
            prompt_or_messages=self._build_messages(prompt),
            model=self.model,
            client=self._client,
            temperature=min(self.temperature, 0.3),  # Lower temp for JSON
            max_tokens=self.max_tokens,
            enable_thinking=False,  # Grammar/json_schema incompatible with thinking
            response_format=schema,
        )

        if result.structured is None:
            raise ValueError(
                "LlamacppGenie.generate_json: No structured result returned. "
                "Ensure the model supports json_schema response format."
            )

        if not result.structured.success:
            errors = result.structured.validation_errors or [
                result.structured.error or "Unknown validation error"
            ]
            raise ValueError(
                f"LlamacppGenie structured output validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        parsed = result.structured.parsed
        if hasattr(parsed, "model_dump"):
            return parsed.model_dump()
        if isinstance(parsed, dict):
            return parsed
        raise ValueError(
            f"LlamacppGenie.generate_json: Unexpected parsed type: {type(parsed)}"
        )

    async def agenerate_json(self, prompt: str, schema: "BaseModel") -> dict[str, Any]:
        """Generate a structured JSON response asynchronously."""
        from jet.adapters.llama_cpp.llm_utils import achat

        result = await achat(
            prompt_or_messages=self._build_messages(prompt),
            model=self.model,
            client=self._async_client,
            temperature=min(self.temperature, 0.3),
            max_tokens=self.max_tokens,
            enable_thinking=False,
            response_format=schema,
        )

        if result.structured is None:
            raise ValueError(
                "LlamacppGenie.agenerate_json: No structured result returned."
            )

        if not result.structured.success:
            errors = result.structured.validation_errors or [
                result.structured.error or "Unknown validation error"
            ]
            raise ValueError(
                f"LlamacppGenie async structured output validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        parsed = result.structured.parsed
        if hasattr(parsed, "model_dump"):
            return parsed.model_dump()
        if isinstance(parsed, dict):
            return parsed
        raise ValueError(
            f"LlamacppGenie.agenerate_json: Unexpected parsed type: {type(parsed)}"
        )

    @classmethod
    def _is_available(cls) -> bool:
        """Check if required dependencies are installed."""
        return (
            importlib.util.find_spec("openai") is not None
            and importlib.util.find_spec("pydantic") is not None
        )

    def __repr__(self) -> str:
        return f"LlamacppGenie(model='{self.model}', base_url='{self.base_url}')"
