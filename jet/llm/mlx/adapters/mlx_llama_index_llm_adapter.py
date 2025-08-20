from typing import List, Optional, Any, Iterator, Union
from llama_index.core.llms.llm import LLM
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
)
from llama_index.core.base.llms.generic_utils import (
    chat_response_to_completion_response,
)
from jet.llm.mlx.base import MLX
from jet.logger import logger


class MLXLlamaIndexLLMAdapter(LLM):
    """LLM wrapper to integrate MLX with llama_index."""

    def __init__(
        self,
        model: str,
        system_prompt: Optional[str] = None,
        with_history: bool = False,
        **kwargs: Any,
    ):
        super().__init__(system_prompt=system_prompt, **kwargs)
        self._mlx = MLX(model=model, with_history=with_history, **kwargs)

    # ---- Completion API ----
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        logger.debug("MLXLlama.complete called with prompt=%s", prompt)
        response = self._mlx.generate(prompt, **kwargs)
        return CompletionResponse(text=response["choices"][0]["text"])

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> Iterator[CompletionResponse]:
        logger.debug("MLXLlama.stream_complete called with prompt=%s", prompt)
        for response in self._mlx.client.stream_generate(prompt=prompt, **kwargs):
            yield CompletionResponse(text=response["choices"][0]["text"])

    # ---- Chat API ----
    def chat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponse:
        logger.debug("MLXLlama.chat called with messages=%s", messages)
        mlx_messages = [{"role": m.role.value, "content": m.content}
                        for m in messages]
        response = self._mlx.chat(
            messages=mlx_messages, system_prompt=self.system_prompt, **kwargs)
        return ChatResponse(
            message=ChatMessage(
                role="assistant", content=response["choices"][0]["message"]["content"])
        )

    def stream_chat(
        self, messages: List[ChatMessage], **kwargs: Any
    ) -> Iterator[ChatResponse]:
        logger.debug("MLXLlama.stream_chat called with messages=%s", messages)
        mlx_messages = [{"role": m.role.value, "content": m.content}
                        for m in messages]
        for response in self._mlx.stream_chat(messages=mlx_messages, system_prompt=self.system_prompt, **kwargs):
            yield ChatResponse(
                message=ChatMessage(
                    role="assistant", content=response["choices"][0]["message"]["content"])
            )

    # ---- Metadata ----
    @property
    def metadata(self):
        return self._mlx.model  # minimal stub, adapt if llama-index requires more
