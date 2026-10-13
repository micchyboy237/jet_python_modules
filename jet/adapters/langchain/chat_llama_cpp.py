"""LangChain Chat Model adapter for llama.cpp using jet observability."""

from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.llama_cpp.llm_utils_observed import achat as jet_achat
from jet.adapters.llama_cpp.llm_utils_observed import chat as jet_chat
from jet.libs.llama_cpp.usage.chat_stream_types import StreamCompletionResult
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field


class ChatLlamaCpp(SimpleChatModel):
    """Chat model wrapper for llama.cpp using jet's observed chat utilities.

    This class integrates llama.cpp models into LangChain by leveraging
    jet's built-in observability and tool handling features.
    """

    model_name: str = Field(default=LLM_MODEL, alias="model")
    """The name of the model to use."""

    base_url: Optional[str] = Field(default=None)
    """Base URL for the llama.cpp server. If None, uses jet default."""

    temperature: float = 0.7
    """Sampling temperature."""

    max_tokens: int = 16384
    """Maximum number of tokens to generate."""

    top_p: float = 0.8
    """Top-p sampling parameter."""

    top_k: int = 20
    """Top-k sampling parameter."""

    repeat_penalty: float = 1.1
    """Repeat penalty."""

    presence_penalty: float = 1.5
    """Presence penalty."""

    frequency_penalty: float = 0.0
    """Frequency penalty."""

    stop: Optional[List[str]] = None
    """Stop sequences."""

    seed: Optional[int] = None
    """Random seed for reproducible generation."""

    enable_thinking: bool = False
    """Enable thinking/reasoning tokens if supported by the model."""

    class Config:
        allow_population_by_field_name = True

    @property
    def _llm_type(self) -> str:
        return "llama_cpp"

    def _convert_messages_to_prompt(
        self, messages: List[BaseMessage]
    ) -> List[Dict[str, Any]]:
        """Convert LangChain messages to OpenAI-compatible message format."""
        formatted_messages = []
        for msg in messages:
            role = "user"
            if isinstance(msg, AIMessage):
                role = "assistant"
            elif hasattr(msg, "type") and msg.type == "system":
                role = "system"

            content = msg.content
            if isinstance(content, list):
                # Handle multimodal content if necessary, simplifying to string for now
                # or extracting text parts
                text_parts = [
                    part["text"]
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                content = "\n".join(text_parts)

            formatted_messages.append({"role": role, "content": content})
        return formatted_messages

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Call out to llama.cpp's chat endpoint."""
        formatted_messages = self._convert_messages_to_prompt(messages)

        # Merge kwargs with instance defaults
        params = {
            "prompt_or_messages": formatted_messages,
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "seed": self.seed,
            "enable_thinking": self.enable_thinking,
        }

        if stop is not None:
            params["stop"] = stop
        elif self.stop is not None:
            params["stop"] = self.stop

        # Pass through any extra kwargs
        params.update(kwargs)

        # Use jet's observed chat function
        result: StreamCompletionResult = jet_chat(**params)

        message = AIMessage(content=result.content)
        generation = ChatGeneration(message=message)

        llm_output = {}
        if result.usage:
            llm_output["token_usage"] = result.usage
        if result.finish_reason:
            llm_output["finish_reason"] = result.finish_reason

        return ChatResult(generations=[generation], llm_output=llm_output)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Asynchronously call out to llama.cpp's chat endpoint."""
        formatted_messages = self._convert_messages_to_prompt(messages)

        params = {
            "prompt_or_messages": formatted_messages,
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "seed": self.seed,
            "enable_thinking": self.enable_thinking,
        }

        if stop is not None:
            params["stop"] = stop
        elif self.stop is not None:
            params["stop"] = self.stop

        params.update(kwargs)

        result: StreamCompletionResult = await jet_achat(**params)

        message = AIMessage(content=result.content)
        generation = ChatGeneration(message=message)

        llm_output = {}
        if result.usage:
            llm_output["token_usage"] = result.usage
        if result.finish_reason:
            llm_output["finish_reason"] = result.finish_reason

        return ChatResult(generations=[generation], llm_output=llm_output)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream chunks from llama.cpp."""
        # Note: jet_chat currently returns a complete result, not a stream iterator.
        # If streaming is required, jet's underlying implementation would need to
        # expose a streaming iterator. For now, we fall back to _generate behavior
        # or simulate streaming if jet supports it.
        # Assuming jet_chat is blocking, we yield the final result as a single chunk
        # or implement proper streaming if jet exposes it.

        # Since jet_chat returns StreamCompletionResult after completion,
        # true streaming via LangChain requires jet to support an on_chunk callback
        # that we can bridge.
        # For compatibility, we'll just yield the final message as one chunk if
        # streaming isn't natively supported by the jet wrapper in a way LangChain expects.

        result = self._generate(messages, stop, run_manager, **kwargs)
        for gen in result.generations:
            yield ChatGenerationChunk(
                message=AIMessageChunk(content=gen.message.content)
            )

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Asynchronously stream chunks from llama.cpp."""
        result = await self._agenerate(messages, stop, run_manager, **kwargs)
        for gen in result.generations:
            yield ChatGenerationChunk(
                message=AIMessageChunk(content=gen.message.content)
            )
