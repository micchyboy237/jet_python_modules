from typing import Any, Dict, List, Optional, Iterator, AsyncIterator
from llama_index.core.llms import LLM, ChatMessage, ChatResponse, CompletionResponse
from llama_index.core.base.llms.types import (
    ChatResponseGen,
    ChatResponseAsyncGen,
    CompletionResponseGen,
    CompletionResponseAsyncGen,
)
from jet.llm.mlx.base import MLX
from jet.llm.mlx.mlx_types import Message
import asyncio


class MLXLlamaIndexLLMAdapter(LLM):
    def __init__(
        self,
        model: str = "qwen3-1.7b-4bit",
        log_dir: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._mlx = MLX(model=model, log_dir=log_dir)

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
        response = self._mlx.chat(messages=mlx_messages, **kwargs)
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
        response = self._mlx.generate(prompt=prompt, **kwargs)
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
        for response in self._mlx.stream_chat(messages=mlx_messages, **kwargs):
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
        for response in self._mlx.stream_generate(prompt=prompt, **kwargs):
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

    # ---- Metadata ----
    @property
    def metadata(self):
        return self._mlx.model  # minimal stub, adapt if llama-index requires more
