from jet.llm.mlx.mlx_utils import has_tools
from jet.llm.mlx.chat_history import ChatHistory
from jet.llm.mlx.remote.utils import prepare_chat_request, process_chat_response, process_stream_chat_response
from jet.llm.mlx.remote.types import ChatCompletionRequest, ChatCompletionResponse
from jet.llm.mlx.remote.client import MLXRemoteClient
from typing import Any, Dict, Sequence, Optional, AsyncGenerator, Union
from autogen_ext.models.ollama import OllamaChatCompletionClient as BaseOllamaChatCompletionClient
from autogen_ext.models.ollama._model_info import _MODEL_INFO
from autogen_core import FunctionCall
from autogen_core.tools import Tool, ToolSchema
from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    RequestUsage,
    ModelInfo,
)
from jet.llm.mlx.logger_utils import ChatLogger
from jet.llm.mlx.config import DEFAULT_OLLAMA_LOG_DIR
from jet.logger import logger
from jet.transformers.formatters import format_json
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
from pathlib import Path

DETERMINISTIC_LLM_SETTINGS = {
    "seed": 42,
    "temperature": 0,
    "max_tokens": -1,
}

# Model mapping for MLX models
MODEL_MAPPING = {
    "llama3.1": "mlx-community/Llama-3.1-8B-Instruct-4bit",
    "llama3.1:8b": "mlx-community/Llama-3.1-8B-Instruct-4bit",
    "llama3.2": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "llama3.2:3b": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "llama3.2:1b": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "deepseek-r1": "mlx-community/DeepSeek-R1-Distill-Qwen-14B-3bit",
    "deepseek-r1:14b": "mlx-community/DeepSeek-R1-Distill-Qwen-14B-3bit",
    "mistral": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "mistral:7b": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "mistral-nemo": "mlx-community/Mistral-Nemo-Instruct-2407-4bit",
    "mistral-nemo:12b": "mlx-community/Mistral-Nemo-Instruct-2407-4bit",
    "qwen2.5": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "qwen2.5:7b": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "qwen3": "mlx-community/Qwen3-8B-4bit-DWQ",
    "qwen3:0.6b": "mlx-community/Qwen3-0.6B-4bit-DWQ",
    "qwen3:1.7b": "mlx-community/Qwen3-1.7B-4bit-DWQ-053125",
    "qwen3:4b": "mlx-community/Qwen3-4B-4bit-DWQ",
    "qwen3:8b": "mlx-community/Qwen3-8B-4bit-DWQ",
}


class MLXChatCompletionClient(BaseOllamaChatCompletionClient):
    def __init__(self, model: str, options: Optional[Dict[str, Any]] = None, base_url: Optional[str] = None, use_remote: bool = True, **kwargs):
        self.use_remote = use_remote
        self.base_url = base_url
        if self.use_remote:
            self._remote_client = MLXRemoteClient(
                base_url=base_url, verbose=kwargs.get("verbose", False))
        else:
            mlx_model = MODEL_MAPPING.get(model.lower(), model)
            options = {**DETERMINISTIC_LLM_SETTINGS, **(options or {})}
            try:
                self._mlx_model, self._tokenizer = load(mlx_model)
            except Exception as e:
                raise ValueError(
                    f"Failed to load MLX model {mlx_model}: {str(e)}")
        model_key = model.lower().split(":")[0]
        if model_key not in _MODEL_INFO:
            model_info = ModelInfo(
                name=mlx_model if not self.use_remote else model,
                family="mlx",
                vision=False,
                function_calling=True,
                json_output=True,
                token_limit=32768,
                structured_output=True,
            )
        else:
            model_info = ModelInfo(
                name=mlx_model if not self.use_remote else model,
                **_MODEL_INFO[model_key]
            )
        create_args = {
            "model": model if self.use_remote else mlx_model,
            **(options or {}),  # Ensure options are included
            **kwargs  # Include any additional kwargs
        }
        super().__init__(
            client=None,
            create_args=create_args,
            model_info=model_info,
            **kwargs
        )
        self._model_name = model if self.use_remote else mlx_model
        self._chat_history = ChatHistory()

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        tool_choice: Union[Tool, str] = "auto",
        json_output: Optional[Union[bool, type]] = None,
        extra_create_args: Dict[str, Any] = {},
    ) -> CreateResult:
        logger.gray("MLX Chat LLM Settings:")
        logger.info(format_json({
            "messages": [m.model_dump() for m in messages],
            "tools": [t.model_dump() if isinstance(t, Tool) else t for t in tools],
            "tool_choice": str(tool_choice),
            "extra_create_args": extra_create_args,
        }))
        create_params = self._process_create_args(
            messages, tools, tool_choice, json_output, extra_create_args
        )
        if self.use_remote:
            req = prepare_chat_request(
                messages=create_params.messages,
                history=self._chat_history,
                system_prompt=None,
                with_history=False,
                response_format="json" if json_output else "text",
                model=self._model_name,
                max_tokens=create_params.create_args.get("max_tokens"),
                temperature=create_params.create_args.get("temperature"),
                seed=create_params.create_args.get("seed"),
                tools=[t for t in tools if isinstance(t, Tool)] if tools and has_tools(
                    self._model_name) else None,
                stream=False
            )
            response = list(
                self._remote_client.create_chat_completion(req, stream=False))[0]
            processed_response = process_chat_response(
                response, self._chat_history, False, tools)
            content = processed_response.get("content", "")
            tool_calls = processed_response.get("tool_calls", None)
            finish_reason = processed_response.get(
                "choices", [{}])[0].get("finish_reason", "stop")
            usage = RequestUsage(
                prompt_tokens=processed_response.get(
                    "usage", {}).get("prompt_tokens", 0),
                completion_tokens=processed_response.get(
                    "usage", {}).get("completion_tokens", 0),
            )
        else:
            prompt = self._messages_to_prompt(create_params.messages)
            response = generate(
                model=self._mlx_model,
                tokenizer=self._tokenizer,
                prompt=prompt,
                max_tokens=create_params.create_args.get("max_tokens", -1),
                temp=create_params.create_args.get("temperature", 0.0),
                seed=create_params.create_args.get("seed", 42),
            )
            prompt_tokens = self.count_tokens(messages, tools=tools)
            completion_tokens = len(self._tokenizer.encode(response))
            usage = RequestUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            content = response
            finish_reason = "stop"
            tool_calls = None
            if tools and tool_choice != "none":
                try:
                    parsed = json.loads(response) if json_output else response
                    if isinstance(parsed, dict) and "tool_calls" in parsed:
                        content = [
                            FunctionCall(
                                id=str(self._tool_id),
                                arguments=json.dumps(call["arguments"]),
                                name=normalize_name(call["name"]),
                            )
                            for call in parsed["tool_calls"]
                        ]
                        finish_reason = "tool_calls"
                        thought = parsed.get("thought")
                        self._tool_id += 1
                except json.JSONDecodeError:
                    pass
        logger.info(
            LLMCallEvent(
                messages=[m.model_dump() for m in create_params.messages],
                response={"content": content},
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
            )
        )
        result = CreateResult(
            finish_reason=normalize_stop_reason(finish_reason),
            content=content,
            usage=usage,
            cached=False,
            logprobs=None,
            thought=thought if self.use_remote else None,
        )
        self._total_usage = _add_usage(self._total_usage, usage)
        self._actual_usage = _add_usage(self._actual_usage, usage)
        return result

    async def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        tool_choice: Union[Tool, str] = "auto",
        json_output: Optional[Union[bool, type]] = None,
        extra_create_args: Dict[str, Any] = {},
        method: str = "stream_chat",
    ) -> AsyncGenerator[Union[str, CreateResult], None]:
        logger.gray("MLX Stream Chat LLM Settings:")
        logger.info(format_json({
            "messages": [m.model_dump() for m in messages],
            "tools": [t.model_dump() if isinstance(t, Tool) else t for t in tools],
            "tool_choice": str(tool_choice),
            "extra_create_args": extra_create_args,
        }))
        create_params = self._process_create_args(
            messages, tools, tool_choice, json_output, extra_create_args
        )
        if self.use_remote:
            req = prepare_chat_request(
                messages=create_params.messages,
                history=self._chat_history,
                system_prompt=None,
                with_history=False,
                response_format="json" if json_output else "text",
                model=self._model_name,
                max_tokens=create_params.create_args.get("max_tokens"),
                temperature=create_params.create_args.get("temperature"),
                seed=create_params.create_args.get("seed"),
                tools=[t for t in tools if isinstance(t, Tool)] if tools and has_tools(
                    self._model_name) else None,
                stream=True
            )
            chunks = self._remote_client.create_chat_completion(
                req, stream=True)
            content_chunks = []
            first_chunk = True
            for chunk in process_stream_chat_response(chunks, self._chat_history, False, tools):
                content = chunk.get("content", "")
                if content:
                    content_chunks.append(content)
                    logger.teal(content, flush=True)
                    yield content
                if first_chunk:
                    first_chunk = False
                    logger.info(
                        LLMStreamStartEvent(
                            messages=[m.model_dump()
                                      for m in create_params.messages],
                        )
                    )
                if chunk.get("choices", [{}])[0].get("finish_reason"):
                    usage = RequestUsage(
                        prompt_tokens=chunk.get(
                            "usage", {}).get("prompt_tokens", 0),
                        completion_tokens=chunk.get(
                            "usage", {}).get("completion_tokens", 0),
                    )
                    finish_reason = chunk.get("choices", [{}])[
                        0].get("finish_reason", "stop")
                    content = "".join(
                        content_chunks) if content_chunks else None
                    tool_calls = chunk.get("tool_calls", None)
                    result = CreateResult(
                        finish_reason=normalize_stop_reason(finish_reason),
                        content=content,
                        usage=usage,
                        cached=False,
                        logprobs=None,
                        thought=chunk.get("thought", None),
                    )
                    logger.info(
                        LLMStreamEndEvent(
                            response=result.model_dump(),
                            prompt_tokens=usage.prompt_tokens,
                            completion_tokens=usage.completion_tokens,
                        )
                    )
                    ChatLogger(DEFAULT_OLLAMA_LOG_DIR, method=method).log_interaction(
                        messages,
                        result.model_dump(),
                        model=self._model_name,
                        tools=tools,
                    )
                    self._total_usage = _add_usage(self._total_usage, usage)
                    self._actual_usage = _add_usage(self._actual_usage, usage)
                    yield result
        else:
            prompt = self._messages_to_prompt(create_params.messages)
            content_chunks = []
            first_chunk = True
            async for chunk in self._stream_local(
                prompt,
                create_params.create_args.get("max_tokens", -1),
                create_params.create_args.get("temperature", 0.0),
                create_params.create_args.get("seed", 42),
            ):
                content_chunks.append(chunk)
                logger.teal(chunk, flush=True)
                if first_chunk:
                    first_chunk = False
                    logger.info(
                        LLMStreamStartEvent(
                            messages=[m.model_dump()
                                      for m in create_params.messages],
                        )
                    )
                yield chunk
            full_content = "".join(content_chunks)
            prompt_tokens = self.count_tokens(messages, tools=tools)
            completion_tokens = len(self._tokenizer.encode(full_content))
            usage = RequestUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            content = full_content
            finish_reason = "stop"
            thought = None
            if tools and tool_choice != "none":
                try:
                    parsed = json.loads(
                        full_content) if json_output else full_content
                    if isinstance(parsed, dict) and "tool_calls" in parsed:
                        content = [
                            FunctionCall(
                                id=str(self._tool_id),
                                arguments=json.dumps(call["arguments"]),
                                name=normalize_name(call["name"]),
                            )
                            for call in parsed["tool_calls"]
                        ]
                        finish_reason = "tool_calls"
                        thought = parsed.get("thought")
                        self._tool_id += 1
                except json.JSONDecodeError:
                    pass
            result = CreateResult(
                finish_reason=normalize_stop_reason(finish_reason),
                content=content,
                usage=usage,
                cached=False,
                logprobs=None,
                thought=thought,
            )
            logger.info(
                LLMStreamEndEvent(
                    response=result.model_dump(),
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                )
            )
            ChatLogger(DEFAULT_OLLAMA_LOG_DIR, method=method).log_interaction(
                messages,
                result.model_dump(),
                model=self._model_name,
                tools=tools,
            )
            self._total_usage = _add_usage(self._total_usage, usage)
            self._actual_usage = _add_usage(self._actual_usage, usage)
            yield result

    async def _stream_local(
        self,
        prompt: str,
        max_tokens: int,
        temp: float,
        seed: int,
    ) -> AsyncGenerator[str, None]:
        for chunk in generate(
            model=self._mlx_model,
            tokenizer=self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=temp,
            seed=seed,
            stream=True,
        ):
            yield chunk
