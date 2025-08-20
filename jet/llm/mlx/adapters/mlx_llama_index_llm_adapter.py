from typing import Any, Dict, List, Optional, Iterator, AsyncIterator
from llama_index.core.llms import LLM, ChatMessage, ChatResponse, CompletionResponse
from llama_index.core.base.llms.types import (
    ChatResponseGen,
    ChatResponseAsyncGen,
    CompletionResponseGen,
    CompletionResponseAsyncGen,
    LLMMetadata,
    MessageRole
)
from jet.llm.mlx.base import MLX
from jet.llm.mlx.mlx_types import Message
from jet.models.model_registry.transformers.mlx_model_registry import MLXModelRegistry
from jet.models.model_types import LLMModelType
from jet.models.utils import get_context_size
import asyncio


class MLXLlamaIndexLLMAdapter(LLM):
    def __init__(
        self,
        model: LLMModelType = "qwen3-1.7b-4bit",
        log_dir: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._mlx = MLXModelRegistry.load_model(model=model, log_dir=log_dir)
        self._metadata = LLMMetadata(
            context_window=get_context_size(self._mlx.model_path),
            num_output=256,  # Default output token limit, adjustable
            is_chat_model=True,  # MLX supports chat functionality
            # MLX does not support function calling by default
            is_function_calling_model=False,
            model_name=model,
            system_role=MessageRole.SYSTEM,  # Standard system role for MLX
        )

    @property
    def metadata(self) -> LLMMetadata:
        return self._metadata

    def chat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponse:
        """
        Synchronous chat implementation.
        Converts llama_index ChatMessages to MLX Message format and uses MLX chat method.
        """
        mlx_messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
            if msg.content is not None
        ]
        generation_settings = {
            "messages": mlx_messages,
            "verbose": True,
            "temperature": 0.3,
            **kwargs,
        }
        response = self._mlx.chat(**generation_settings)
        # Assuming response is a CompletionResponse with a choices list
        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})
        return ChatResponse(
            message=ChatMessage(
                role="assistant",
                content=message.get("content", ""),
            ),
            raw=response,
        )

    async def achat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponse:
        """
        Asynchronous chat implementation.
        Wraps the synchronous chat method in an async context.
        """
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: self.chat(messages, **kwargs)
        )
        return result

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        """
        Synchronous completion implementation.
        Uses MLX generate method for text completion.
        """
        generation_settings = {
            "prompt": prompt,
            "verbose": True,
            "temperature": 0.3,
            **kwargs,
        }
        response = self._mlx.generate(**generation_settings)
        # Assuming response is a CompletionResponse with a choices list
        choice = response.get("choices", [{}])[0]
        return CompletionResponse(
            text=choice.get("text", ""),
            raw=response,
        )

    async def acomplete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        """
        Asynchronous completion implementation.
        Wraps the synchronous complete method in an async context.
        """
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: self.complete(prompt, formatted, **kwargs)
        )
        return result

    def stream_chat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        """
        Synchronous streaming chat implementation.
        Converts llama_index ChatMessages to MLX Message format and uses MLX stream_chat.
        """
        mlx_messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
            if msg.content is not None
        ]
        generation_settings = {
            "messages": mlx_messages,
            "verbose": True,
            "temperature": 0.3,
            **kwargs,
        }
        for response in self._mlx.stream_chat(**generation_settings):
            choice = response.get("choices", [{}])[0]
            message = choice.get("message", {})
            yield ChatResponse(
                message=ChatMessage(
                    role="assistant",
                    content=message.get("content", ""),
                ),
                delta=message.get("content", ""),
                raw=response,
            )

    async def astream_chat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponseAsyncGen:
        """
        Asynchronous streaming chat implementation.
        Wraps the synchronous stream_chat method in an async context.
        """
        def sync_iterator():
            return self.stream_chat(messages, **kwargs)

        async def async_generator() -> ChatResponseAsyncGen:
            loop = asyncio.get_event_loop()
            sync_gen = sync_iterator()
            while True:
                try:
                    result = await loop.run_in_executor(None, lambda: next(sync_gen))
                    yield result
                except StopIteration:
                    break

        return async_generator()

    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        """
        Synchronous streaming completion implementation.
        Uses MLX stream_generate method for streaming text completion.
        """
        generation_settings = {
            "prompt": prompt,
            "verbose": True,
            "temperature": 0.3,
            **kwargs,
        }
        for response in self._mlx.stream_generate(**generation_settings):
            choice = response.get("choices", [{}])[0]
            yield CompletionResponse(
                text=choice.get("text", ""),
                delta=choice.get("text", ""),
                raw=response,
            )

    async def astream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseAsyncGen:
        """
        Asynchronous streaming completion implementation.
        Wraps the synchronous stream_complete method in an async context.
        """
        def sync_iterator():
            return self.stream_complete(prompt, formatted, **kwargs)

        async def async_generator() -> CompletionResponseAsyncGen:
            loop = asyncio.get_event_loop()
            sync_gen = sync_iterator()
            while True:
                try:
                    result = await loop.run_in_executor(None, lambda: next(sync_gen))
                    yield result
                except StopIteration:
                    break

        return async_generator()
