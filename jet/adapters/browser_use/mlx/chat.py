from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Iterator, List, Optional, TypeVar, Union, overload
from jet.llm.mlx.remote.client import MLXRemoteClient
from jet.llm.mlx.remote.generation import stream_chat
from jet.llm.mlx.remote.types import ChatCompletionResponse
from jet.llm.mlx.config import DEFAULT_LOG_DIR
from browser_use.llm.base import BaseChatModel
from browser_use.llm.exceptions import ModelProviderError
from browser_use.llm.messages import BaseMessage
from browser_use.llm.views import ChatInvokeCompletion
from jet.adapters.browser_use.mlx.serializer import MLXMessageSerializer
from jet.logger import logger
from pydantic import BaseModel
from typing import Callable, Dict, Literal

from jet.models.model_types import LLMModelValue

T = TypeVar('T', bound=BaseModel)


@dataclass
class ChatMLX(BaseChatModel):
    """
    A wrapper around MLX's chat model for browser_use.
    """
    model: LLMModelValue = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    log_dir: str = DEFAULT_LOG_DIR
    base_url: Optional[str] = None
    verbose: bool = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    repetition_context_size: Optional[int] = None
    logit_bias: Optional[Dict[int, float]] = None
    logprobs: Optional[int] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    tools: Optional[List[Callable]] = None
    response_format: Optional[Union[Literal["text",
                                            "json"], Dict[str, Any]]] = None

    @property
    def provider(self) -> str:
        return 'mlx'

    def get_client(self) -> MLXRemoteClient:
        """Returns an MLXRemoteClient instance."""
        return MLXRemoteClient(base_url=self.base_url, verbose=self.verbose)

    @property
    def name(self) -> str:
        return self.model

    @overload
    async def ainvoke(self, messages: List[BaseMessage],
                      output_format: None = None) -> ChatInvokeCompletion[str]: ...

    @overload
    async def ainvoke(self, messages: List[BaseMessage],
                      output_format: type[T]) -> ChatInvokeCompletion[T]: ...

    async def ainvoke(
        self, messages: List[BaseMessage], output_format: type[T] | None = None
    ) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
        """Invoke a non-streaming chat completion."""
        try:
            mlx_messages = MLXMessageSerializer.serialize_messages(messages)
            response_stream = stream_chat(
                messages=mlx_messages,
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                repetition_penalty=self.repetition_penalty,
                repetition_context_size=self.repetition_context_size,
                logit_bias=self.logit_bias,
                logprobs=self.logprobs,
                seed=self.seed,
                stop=self.stop,
                tools=self.tools,
                # client=self.get_client(),
                # base_url=self.base_url,
                base_url="http://localhost:8080",
                # verbose=self.verbose,
                verbose=True,
                response_format=self.response_format if output_format is None else output_format.model_json_schema(),
            )
            response_text = ""
            for chunk in response_stream:  # Use async for to iterate over async generator
                content = chunk["choices"][0]["message"]["content"]
                response_text += content

            completion = response_text
            if output_format is not None and completion:
                completion = output_format.model_validate_json(completion)
            return ChatInvokeCompletion(completion=completion, usage=response.get("usage"))
        except Exception as e:
            raise ModelProviderError(message=str(e), model=self.name) from e
