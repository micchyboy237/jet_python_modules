import os
import json
import logging
from logging.handlers import RotatingFileHandler
from typing import List, Sequence
from datetime import datetime
from autogen_core.models import LLMMessage, SystemMessage, UserMessage, AssistantMessage, FunctionExecutionResult, FunctionExecutionResultMessage
from autogen_core.tools import Tool, ToolSchema
from autogen_ext.models.ollama import OllamaChatCompletionClient as BaseOllamaChatCompletionClient
from autogen_ext.models.ollama._ollama_client import convert_tools
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    ImageBlock,
    LLMMetadata,
    MessageRole,
    TextBlock,
)
from jet._token.token_utils import token_counter
from jet.transformers.object import make_serializable


class OllamaChatCompletionClient(BaseOllamaChatCompletionClient):
    def __init__(self, *args, log_dir: str | None = None, **kwargs):
        """
        Extension of BaseOllamaChatCompletionClient with optional request/response logging.

        Args:
            log_dir (str | None): Directory to store request/response logs. If None, logging is disabled.
        """
        super().__init__(*args, **kwargs)

        self._log_dir = log_dir
        self._logger = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

            log_path = os.path.join(log_dir, "ollama_client.log")
            handler = RotatingFileHandler(
                log_path, maxBytes=5 * 1024 * 1024, backupCount=5)
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)

            self._logger = logging.getLogger(f"OllamaClientLogger_{id(self)}")
            self._logger.setLevel(logging.INFO)
            self._logger.addHandler(handler)
            self._logger.propagate = False

    def _log_request(self, endpoint: str, payload: dict):
        if not self._logger:
            return
        try:
            self._logger.info(
                "REQUEST %s\n%s",
                endpoint,
                json.dumps(payload, indent=2, ensure_ascii=False),
            )
        except Exception as e:
            self._logger.error(f"Failed to log request: {e}")

    def _log_response(self, endpoint: str, response: dict):
        if not self._logger:
            return
        try:
            self._logger.info(
                "RESPONSE %s\n%s",
                endpoint,
                json.dumps(response, indent=2, ensure_ascii=False),
            )
        except Exception as e:
            self._logger.error(f"Failed to log response: {e}")

    async def create(self, *args, **kwargs):
        # Log request before sending
        self._log_request("create", {"args": args, "kwargs": kwargs})

        result = await super().create(*args, **kwargs)

        # Log response after receiving
        try:
            self._log_response("create", result.model_dump())
        except Exception as e:
            if self._logger:
                self._logger.error(f"Failed to log response: {e}")

        return result

    async def create_stream(self, *args, **kwargs):
        # Log request
        self._log_request("create_stream", {"args": args, "kwargs": kwargs})

        async for chunk in super().create_stream(*args, **kwargs):
            try:
                if hasattr(chunk, "model_dump"):
                    self._log_response("create_stream", chunk.model_dump())
                else:
                    self._log_response("create_stream", {"chunk": str(chunk)})
            except Exception as e:
                if self._logger:
                    self._logger.error(f"Failed to log streaming chunk: {e}")
            yield chunk

    def count_tokens(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
    ) -> int:
        """
        Count tokens for a sequence of LLMMessage objects, handling all supported message types.
        Also counts tokens for Ollama tools.
        """
        ollama_messages = self._convert_to_ollama_messages(messages)
        ollama_tools = convert_tools(tools)
        ollama_tools_dict = make_serializable(ollama_tools)
        msg_token_count: int = token_counter(
            ollama_messages, self._create_args["model"])
        tools_token_count: int = token_counter(
            ollama_tools_dict, self._create_args["model"]) if ollama_tools else 0
        return msg_token_count + tools_token_count

    def _convert_to_ollama_messages(self, messages: Sequence[LLMMessage]) -> List[ChatMessage]:
        """
        Convert LLMMessage objects to llama_index ChatMessage objects, handling all supported message types.
        Applies correct role mapping for function/tool messages.
        """
        ollama_messages: List[ChatMessage] = []
        for message in messages:
            if isinstance(message, SystemMessage):
                ollama_messages.append(
                    ChatMessage(role="system", content=getattr(
                        message, "content", ""))
                )
            elif isinstance(message, UserMessage):
                content = getattr(message, "content", "")
                if isinstance(content, list):
                    text_content = " ".join(str(x) for x in content)
                else:
                    text_content = str(content)
                ollama_messages.append(
                    ChatMessage(role="user", content=text_content)
                )
            elif isinstance(message, AssistantMessage):
                content = getattr(message, "content", "")
                if isinstance(content, list):
                    text_content = " ".join(str(x) for x in content)
                else:
                    text_content = str(content)
                ollama_messages.append(
                    ChatMessage(role="assistant", content=text_content)
                )
            elif isinstance(message, FunctionExecutionResultMessage):
                # Each FunctionExecutionResult in content
                content = getattr(message, "content", [])
                for result in content:
                    # Use "tool" role for function execution results, as per OpenAI/llama_index conventions
                    ollama_messages.append(
                        ChatMessage(role="tool", content=str(
                            getattr(result, "content", "")))
                    )
            else:
                # Fallback: treat as user message
                ollama_messages.append(
                    ChatMessage(role="user", content=str(
                        getattr(message, "content", "")))
                )
        return ollama_messages
