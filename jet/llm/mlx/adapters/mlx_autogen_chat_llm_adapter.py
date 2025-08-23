import os
from typing import Any, AsyncGenerator, Dict, Iterator, List, Literal, Mapping, Optional, Sequence, Union, cast
from pydantic import BaseModel
from autogen_core.models import ChatCompletionClient, CreateResult, LLMMessage, ModelInfo, RequestUsage, ModelCapabilities, FinishReasons
from autogen_core.tools import Tool, ToolSchema
from autogen_core import CancellationToken
from jet.file.utils import save_file
from jet.llm.mlx.base import MLX
from jet.llm.mlx.client import CompletionResponse, Message
from jet.models.model_registry.transformers.mlx_model_registry import MLXModelRegistry
from jet.models.model_types import LLMModelType
from jet.logger import logger
from jet.db.postgres.config import DEFAULT_HOST, DEFAULT_PASSWORD, DEFAULT_PORT, DEFAULT_USER

DEFAULT_DB = "mlx_agents_chat_history_db1"


class MLXAutogenChatLLMAdapter(ChatCompletionClient):
    """A chat completion client for MLX models, compatible with AutoGen's ChatCompletionClient."""

    def __init__(
        self,
        model: LLMModelType,
        adapter_path: Optional[str] = None,
        draft_model: Optional[LLMModelType] = None,
        trust_remote_code: bool = False,
        chat_template: Optional[str] = None,
        use_default_chat_template: bool = True,
        dbname: str = DEFAULT_DB,
        user: str = DEFAULT_USER,
        password: str = DEFAULT_PASSWORD,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        overwrite_db: bool = False,
        session_id: Optional[str] = None,
        with_history: bool = True,
        seed: Optional[int] = None,
        log_dir: Optional[str] = None,
        device: Optional[Literal["cpu", "mps"]] = "mps",
        temperature: float = 0.0,
    ):
        """Initialize the MLX chat completion client."""
        super().__init__()
        self.client = MLXModelRegistry.load_model(
            model=model,
            adapter_path=adapter_path,
            draft_model=draft_model,
            trust_remote_code=trust_remote_code,
            chat_template=chat_template,
            use_default_chat_template=use_default_chat_template,
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port,
            overwrite_db=overwrite_db,
            session_id=session_id,
            with_history=with_history,
            seed=seed,
            log_dir=log_dir,
            device=device,
            temperature=temperature,
        )
        self._usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self.log_dir = log_dir

    def _save_logs(self, args_dict: Dict) -> None:
        if self.log_dir:
            autogen_dir = os.path.join(self.log_dir, "create_args")
            os.makedirs(autogen_dir, exist_ok=True)
            existing_files = [f for f in os.listdir(
                autogen_dir) if f.endswith(".json")]
            numbers = []
            for fname in existing_files:
                try:
                    num = int(fname.split(".")[0])
                    numbers.append(num)
                except Exception:
                    continue
            next_number = max(numbers) + 1 if numbers else 1
            incremented_filename = f"{next_number}"
            save_file(
                args_dict, f"{autogen_dir}/args_{incremented_filename}.json")

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
        from autogen_core.models import SystemMessage, UserMessage, AssistantMessage
        self._save_logs({
            "stream": False,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "json_output": json_output,
            "extra_create_args": extra_create_args,
            "cancellation_token": cancellation_token,
        })
        mlx_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name if isinstance(tool, Tool) else tool.get("name", ""),
                    "description": tool.description if isinstance(tool, Tool) else tool.get("description", ""),
                    "parameters": tool.schema if isinstance(tool, Tool) else tool.get("parameters", {}),
                },
            }
            for tool in tools
        ]
        mlx_messages = [
            {
                "role": (
                    "system" if isinstance(msg, SystemMessage) else
                    "user" if isinstance(msg, UserMessage) else
                    "assistant"
                ),
                "content": msg.content
            }
            for msg in messages
        ]
        system_prompt = next(
            (msg["content"]
             for msg in mlx_messages if msg["role"] == "system"), None
        )
        if system_prompt:
            mlx_messages = [
                msg for msg in mlx_messages if msg["role"] != "system"]

        create_args = {
            "messages": mlx_messages,
            "tools": mlx_tools if mlx_tools else None,
            "system_prompt": system_prompt,
            "max_tokens": extra_create_args.get("max_tokens", -1),
            "temperature": extra_create_args.get("temperature", 0.0),
            "top_p": extra_create_args.get("top_p", 1.0),
            "stop": extra_create_args.get("stop", None),
            "logprobs": extra_create_args.get("logprobs", -1),
            "repetition_penalty": extra_create_args.get("repetition_penalty", None),
            "repetition_context_size": extra_create_args.get("repetition_context_size", 20),
            "log_dir": extra_create_args.get("log_dir", None),
            "verbose": extra_create_args.get("verbose", True),
        }
        response = self.client.chat(**create_args)
        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "")
        tool_calls = message.get("tool_calls", None)
        function_calls = []
        if tool_calls:
            for call in tool_calls:
                function = call.get("function", {})
                function_calls.append(
                    {
                        # "id": call.get("id", ""),
                        "function": {
                            "name": function.get("name", ""),
                            "arguments": function.get("arguments", ""),
                        },
                        "type": "function",
                    }
                )
        prompt_tokens = self.client.count_tokens(messages)
        completion_tokens = self.client.count_tokens(content)
        self._usage = RequestUsage(
            prompt_tokens=self._usage.prompt_tokens + prompt_tokens,
            completion_tokens=self._usage.completion_tokens + completion_tokens,
        )
        finish_reason: FinishReasons = "function_calls" if function_calls else "stop"
        return CreateResult(
            content=content,
            usage=RequestUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ),
            finish_reason=finish_reason,
            cached=False,
            logprobs=None,
            thought=None,
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
        from autogen_core.models import SystemMessage, UserMessage, AssistantMessage
        self._save_logs({
            "stream": True,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "json_output": json_output,
            "extra_create_args": extra_create_args,
            "cancellation_token": cancellation_token,
        })
        mlx_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name if isinstance(tool, Tool) else tool.get("name", ""),
                    "description": tool.description if isinstance(tool, Tool) else tool.get("description", ""),
                    "parameters": tool.schema if isinstance(tool, Tool) else tool.get("parameters", {}),
                },
            }
            for tool in tools
        ]
        mlx_messages = [
            {
                "role": (
                    "system" if isinstance(msg, SystemMessage) else
                    "user" if isinstance(msg, UserMessage) else
                    "assistant"
                ),
                "content": msg.content
            }
            for msg in messages
        ]
        system_prompt = next(
            (msg["content"]
             for msg in mlx_messages if msg["role"] == "system"), None
        )
        if system_prompt:
            mlx_messages = [
                msg for msg in mlx_messages if msg["role"] != "system"]

        create_args = {
            "messages": mlx_messages,
            "tools": mlx_tools if mlx_tools else None,
            "system_prompt": system_prompt,
            "max_tokens": extra_create_args.get("max_tokens", -1),
            "temperature": extra_create_args.get("temperature", 0.0),
            "top_p": extra_create_args.get("top_p", 1.0),
            "stop": extra_create_args.get("stop", None),
            "logprobs": extra_create_args.get("logprobs", -1),
            "repetition_penalty": extra_create_args.get("repetition_penalty", None),
            "repetition_context_size": extra_create_args.get("repetition_context_size", 20),
            "log_dir": extra_create_args.get("log_dir", None),
            "verbose": extra_create_args.get("verbose", True),
        }
        prompt_tokens = self.client.count_tokens(messages)
        completion_tokens = 0
        content = ""
        for response in self.client.stream_chat(**create_args):
            choice = response.get("choices", [{}])[0]
            message = choice.get("message", {})
            chunk_content = message.get("content", "")
            content += chunk_content
            completion_tokens += self.client.count_tokens(chunk_content)
            yield chunk_content
        tool_calls = message.get("tool_calls", None)
        function_calls = []
        if tool_calls:
            for call in tool_calls:
                function = call.get("function", {})
                function_calls.append(
                    {
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
        finish_reason: FinishReasons = "function_calls" if function_calls else "stop"
        yield CreateResult(
            content=content,
            function_calls=function_calls,
            usage=RequestUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ),
            finish_reason=finish_reason,
            cached=False,
            logprobs=None,
            thought=None,
        )

    async def close(self) -> None:
        """Close the client and clear history."""
        self.client.reset_model()
        logger.info("MLXAutogenChatLLMAdapter closed.")

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
        return self.client.count_tokens(mlx_messages) + self.client.count_tokens(tool_content)

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        """Calculate remaining tokens."""
        used_tokens = self.count_tokens(messages, tools=tools)
        return self.client.get_remaining_tokens(messages) - used_tokens

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
