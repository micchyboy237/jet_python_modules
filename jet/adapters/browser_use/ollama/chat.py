from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeVar, overload

import httpx
from ollama import AsyncClient as OllamaAsyncClient
from ollama import Options
from pydantic import BaseModel

from browser_use.llm.base import BaseChatModel
from browser_use.llm.exceptions import ModelProviderError
from browser_use.llm.messages import BaseMessage
from browser_use.llm.ollama.serializer import OllamaMessageSerializer
from browser_use.llm.views import ChatInvokeCompletion

from jet.llm.mlx.config import DEFAULT_OLLAMA_LOG_DIR
from jet.llm.mlx.logger_utils import ChatLogger
from jet.logger import logger
from jet._token.token_utils import token_counter
from jet.transformers.formatters import format_json

T = TypeVar('T', bound=BaseModel)


@dataclass
class ChatOllama(BaseChatModel):
    """
    A wrapper around Ollama's chat model.
    """

    model: str
    log_dir: str = DEFAULT_OLLAMA_LOG_DIR

    # # Model params
    # TODO (matic): Why is this commented out?
    # temperature: float | None = None

    # Client initialization parameters
    host: str | None = None
    timeout: float | httpx.Timeout | None = 300.0
    client_params: dict[str, Any] | None = None
    ollama_options: Mapping[str, Any] | Options | None = None

    # Static
    @property
    def provider(self) -> str:
        return 'ollama'

    def _get_client_params(self) -> dict[str, Any]:
        """Prepare client parameters dictionary."""
        return {
            'host': self.host,
            'timeout': self.timeout,
            'client_params': self.client_params,
        }

    def get_client(self) -> OllamaAsyncClient:
        """
        Returns an OllamaAsyncClient client.
        """
        return OllamaAsyncClient(host=self.host, timeout=self.timeout, **self.client_params or {})

    @property
    def name(self) -> str:
        return self.model

    @overload
    async def ainvoke(self, messages: list[BaseMessage],
                      output_format: None = None) -> ChatInvokeCompletion[str]: ...

    @overload
    async def ainvoke(self, messages: list[BaseMessage],
                      output_format: type[T]) -> ChatInvokeCompletion[T]: ...

    async def ainvoke(
        self, messages: list[BaseMessage], output_format: type[T] | None = None
    ) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
        ollama_messages = OllamaMessageSerializer.serialize_messages(messages)

        logger.gray("LLM Settings:")
        logger.info(format_json(self.ollama_options))

        logger.debug(
            f"Prompt Tokens: {token_counter(ollama_messages, self.model)}")

        try:
            if output_format is None:
                response_stream = await self.get_client().chat(
                    stream=True,
                    model=self.model,
                    messages=ollama_messages,
                    options=self.ollama_options,
                )
                response_text = ""
                async for chunk in response_stream:  # Use async for to iterate over async generator
                    content = chunk.message.content
                    logger.teal(content, flush=True)
                    response_text += content

                ChatLogger(self.log_dir, method="chat").log_interaction(
                    ollama_messages,
                    response_text,
                    model=self.model,
                    options=self.ollama_options,
                )

                return ChatInvokeCompletion(completion=response_text, usage=None)
            else:
                schema = output_format.model_json_schema()

                response_stream = await self.get_client().chat(
                    stream=True,
                    model=self.model,
                    messages=ollama_messages,
                    format=schema,
                    options=self.ollama_options,
                )
                response_text = ""
                async for chunk in response_stream:  # Use async for to iterate over async generator
                    content = chunk.message.content
                    logger.teal(content, flush=True)
                    response_text += content

                ChatLogger(self.log_dir, method="chat").log_interaction(
                    ollama_messages,
                    response_text,
                    model=self.model,
                    format=schema,
                    options=self.ollama_options,
                )

                completion = response_text
                if output_format is not None:
                    completion = output_format.model_validate_json(completion)

                return ChatInvokeCompletion(completion=completion, usage=None)

        except Exception as e:
            raise ModelProviderError(message=str(e), model=self.name) from e
