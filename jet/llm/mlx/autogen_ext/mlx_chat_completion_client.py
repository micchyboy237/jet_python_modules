from typing import Any, AsyncGenerator, Iterator, Mapping, Optional, Sequence, Union, cast
from pydantic import BaseModel
from autogen_core.models import ChatCompletionClient, CreateResult, LLMMessage, ModelInfo, RequestUsage, ModelCapabilities
from autogen_core.tools import Tool, ToolSchema
from autogen_core import CancellationToken
from jet.llm.mlx.base import MLX
from jet.llm.mlx.client import CompletionResponse, Message
from jet.models.model_types import LLMModelType
from jet.logger import logger


class MLXChatCompletionClient(ChatCompletionClient):
    """A chat completion client for MLX models, compatible with AutoGen's ChatCompletionClient."""

    def __init__(
        self,
        model: LLMModelType,
        adapter_path: Optional[str] = None,
        draft_model: Optional[LLMModelType] = None,
        trust_remote_code: bool = False,
        chat_template: Optional[str] = None,
        use_default_chat_template: bool = True,
        with_history: bool = True,
        seed: Optional[int] = None,
        device: Optional[str] = "mps",
        log_dir: Optional[str] = None,
    ):
        """Initialize the MLX chat completion client."""
        super().__init__()
        self.mlx_client = MLX(
            model=model,
            adapter_path=adapter_path,
            draft_model=draft_model,
            trust_remote_code=trust_remote_code,
            chat_template=chat_template,
            use_default_chat_template=use_default_chat_template,
            with_history=with_history,
            seed=seed,
            device=device,
            log_dir=log_dir,
        )
        self._usage = RequestUsage(
            prompt_tokens=0, completion_tokens=0)

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        tool_choice: Union[Tool, str] = "auto",
        json_output: Optional[Union[bool, type[BaseModel]]] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> CreateResult:
        """Create a single chat completion response."""
        mlx_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name if isinstance(tool, Tool) else tool["function"]["name"],
                    "description": tool.description if isinstance(tool, Tool) else tool["function"]["description"],
                    "parameters": tool.schema if isinstance(tool, Tool) else tool["function"]["parameters"],
                },
            }
            for tool in tools
        ]
        mlx_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]
        system_prompt = next(
            (msg["content"]
             for msg in mlx_messages if msg["role"] == "system"), None
        )
        if system_prompt:
            mlx_messages = [
                msg for msg in mlx_messages if msg["role"] != "system"]

        # Extract supported parameters from extra_create_args
        supported_args = {
            "logprobs": extra_create_args.get("logprobs"),
            "logit_bias": extra_create_args.get("logit_bias"),
            "repetition_penalty": extra_create_args.get("repetition_penalty"),
            "repetition_context_size": extra_create_args.get("repetition_context_size"),
            "temperature": extra_create_args.get("temperature"),
            "stop": extra_create_args.get("stop"),
            "role_mapping": extra_create_args.get("role_mapping"),
            "verbose": extra_create_args.get("verbose"),
        }
        # Remove None values
        supported_args = {k: v for k,
                          v in supported_args.items() if v is not None}

        # Warn if model parameter is passed
        if "model" in extra_create_args:
            logger.warning(
                "Model parameter is set during MLXChatCompletionClient initialization and cannot be overridden. Ignoring provided model argument.")

        response = self.mlx_client.chat(
            messages=mlx_messages,
            tools=mlx_tools if mlx_tools else None,
            system_prompt=system_prompt,
            **supported_args,
        )
        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "")
        tool_calls = message.get("tool_calls", None)
        finish_reason = choice.get("finish_reason", None)
        function_calls = []
        if tool_calls:
            for call in tool_calls:
                function = call.get("function", {})
                function_calls.append(
                    {
                        "id": call.get("id", ""),
                        "function": {
                            "name": function.get("name", ""),
                            "arguments": function.get("arguments", ""),
                        },
                        "type": "function",
                    }
                )
        prompt_tokens = self.mlx_client.count_tokens(messages)
        completion_tokens = self.mlx_client.count_tokens(content)
        self._usage = RequestUsage(
            prompt_tokens=self._usage.prompt_tokens + prompt_tokens,
            completion_tokens=self._usage.completion_tokens + completion_tokens,
        )
        return CreateResult(
            content=content,
            function_calls=function_calls,
            usage=RequestUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ),
            finish_reason=finish_reason,
        )

    async def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        tool_choice: Union[Tool, str] = "auto",
        json_output: Optional[Union[bool, type[BaseModel]]] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncGenerator[Union[str, CreateResult], None]:
        """Create a stream of chat completion responses."""
        mlx_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name if isinstance(tool, Tool) else tool["function"]["name"],
                    "description": tool.description if isinstance(tool, Tool) else tool["function"]["description"],
                    "parameters": tool.schema if isinstance(tool, Tool) else tool["function"]["parameters"],
                },
            }
            for tool in tools
        ]
        mlx_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]
        system_prompt = next(
            (msg["content"]
             for msg in mlx_messages if msg["role"] == "system"), None
        )
        if system_prompt:
            mlx_messages = [
                msg for msg in mlx_messages if msg["role"] != "system"]

        # Extract supported parameters from extra_create_args
        supported_args = {
            "logprobs": extra_create_args.get("logprobs"),
            "logit_bias": extra_create_args.get("logit_bias"),
            "repetition_penalty": extra_create_args.get("repetition_penalty"),
            "repetition_context_size": extra_create_args.get("repetition_context_size"),
            "temperature": extra_create_args.get("temperature"),
            "stop": extra_create_args.get("stop"),
            "role_mapping": extra_create_args.get("role_mapping"),
            "verbose": extra_create_args.get("verbose"),
        }
        # Remove None values
        supported_args = {k: v for k,
                          v in supported_args.items() if v is not None}

        # Warn if model parameter is passed
        if "model" in extra_create_args:
            logger.warning(
                "Model parameter is set during MLXChatCompletionClient initialization and cannot be overridden. Ignoring provided model argument.")

        prompt_tokens = self.mlx_client.count_tokens(messages)
        completion_tokens = 0
        content = ""
        tool_calls = None
        finish_reason = None
        for response in self.mlx_client.stream_chat(
            messages=mlx_messages,
            tools=mlx_tools if mlx_tools else None,
            system_prompt=system_prompt,
            **supported_args,
        ):
            choice = response.get("choices", [{}])[0]
            message = choice.get("message", {})
            chunk_content = message.get("content", "")
            content += chunk_content
            completion_tokens += self.mlx_client.count_tokens(chunk_content)
            tool_calls = message.get("tool_calls", None)
            finish_reason = choice.get("finish_reason", None)
            yield chunk_content

        function_calls = []
        if tool_calls:
            for call in tool_calls:
                function = call.get("function", {})
                function_calls.append(
                    {
                        "id": call.get("id", ""),
                        "function": {
                            "name": function.get("name", ""),
                            "arguments": function.get("arguments", ""),
                        },
                        "type": "function",
                    }
                )
        self._usage = RequestUsage(
            prompt_tokens=self._usage.prompt_tokens + prompt_tokens,
            completion_tokens=self._usage.completion_tokens + completion_tokens,
        )
        yield CreateResult(
            content=content,
            function_calls=function_calls,
            usage=RequestUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ),
            finish_reason=finish_reason,
        )

    async def close(self) -> None:
        """Close the client and clear history."""
        self.mlx_client.clear_history()
        logger.info("MLXChatCompletionClient closed.")

    def actual_usage(self) -> RequestUsage:
        """Return the actual usage for the last request."""
        return self._usage

    def total_usage(self) -> RequestUsage:
        """Return the total usage across all requests."""
        return self._usage

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        """Count tokens for messages and tools."""
        mlx_messages = [{"role": msg.role, "content": msg.content}
                        for msg in messages]
        tool_content = " ".join(
            tool.description + str(tool.schema) if isinstance(
                tool, Tool) else tool["function"]["description"] + str(tool["function"]["parameters"])
            for tool in tools
        )
        return self.mlx_client.count_tokens(mlx_messages) + self.mlx_client.count_tokens(tool_content)

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        """Calculate remaining tokens."""
        used_tokens = self.count_tokens(messages, tools=tools)
        return self.mlx_client.get_remaining_tokens(messages) - used_tokens

    @property
    def model_info(self) -> ModelInfo:
        """Return model information."""
        return {
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "llama-3.3-8b",
            "structured_output": False,
            "multiple_system_messages": True,
        }

    @property
    def capabilities(self) -> ModelCapabilities:
        """Return model capabilities for compatibility with deprecated interface."""
        return {
            "vision": self.model_info["vision"],
            "function_calling": self.model_info["function_calling"],
            "json_output": self.model_info["json_output"],
        }
