from jet.llm.mlx.remote.types import ChatCompletionResponse
from typing import AsyncIterator, List, Optional, Sequence, Dict, Any
from autogen_ext.models.ollama import OllamaChatCompletionClient as BaseOllamaChatCompletionClient
from autogen_core import (
    FunctionCall,
)
from autogen_core.models import (
    AssistantMessage,
    CreateResult,
    FunctionExecutionResultMessage,
    LLMMessage,
    ModelFamily,
    ModelInfo,
    RequestUsage,
    SystemMessage,
    UserMessage,
)
from jet.llm.logger_utils import ChatLogger
from jet.llm.mlx.config import DEFAULT_OLLAMA_LOG_DIR
from jet.llm.mlx.remote import generation as gen
from jet.llm.mlx.remote.types import ToolCall as MLXToolCall
from jet.logger import logger
from jet.transformers.formatters import format_json

DETERMINISTIC_LLM_SETTINGS = {
    "seed": 42,
    "temperature": 0,
    "num_keep": 0,
    "num_predict": 1024,
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
    def __init__(self, model: str, host: str = "http://localhost:11434", timeout: float = 300.0, options: dict = None, **kwargs):
        mapped_model = MODEL_MAPPING.get(model, model)
        model_info = ModelInfo(
            family=ModelFamily.R1 if "r1" in model.lower() else ModelFamily.UNKNOWN,
            function_calling=True,
            json_output=True,
            vision=False,
            structured_output=True
        )
        options = {**DETERMINISTIC_LLM_SETTINGS, **(options or {})}
        super().__init__(model=model, host=host, timeout=timeout,
                         options=options, model_info=model_info, **kwargs)
        self._model_name = model
        self._host = host
        self._options = options

    async def create(self, messages: Sequence[LLMMessage], **kwargs) -> CreateResult:
        logger.gray("Chat LLM Settings:")
        logger.info(format_json({
            "messages": [msg.model_dump() for msg in messages],
            "kwargs": kwargs,
        }))
        result = None
        async for chunk in self.create_stream(messages, method="chat", **kwargs):
            result = chunk
        return result

    async def create_stream(self, messages: Sequence[LLMMessage], method: str = "stream_chat", **kwargs) -> AsyncIterator[CreateResult]:
        logger.gray("Stream Chat LLM Settings:")
        logger.info(format_json({
            "messages": [msg.model_dump() for msg in messages],
            "kwargs": kwargs,
        }))
        mlx_messages = convert_ollama_to_mlx(messages)
        tools = kwargs.get("tools", [])
        model = MODEL_MAPPING.get(self._model_name, self._model_name)
        max_tokens = self._options["num_predict"] if "num_predict" in self._options else 512
        max_tokens = max_tokens if max_tokens > 0 else 512
        async for chunk in gen.astream_chat(
            messages=mlx_messages,
            model=model,
            tools=tools,
            max_tokens=max_tokens,
            temperature=self._options.get("temperature", 0),
            seed=self._options.get("seed", 42),
            verbose=True,
            base_url=self._host,
        ):
            content = ""
            if isinstance(chunk, dict) and "choices" in chunk:
                create_result = self._convert_mlx_to_create_result(chunk)
                content += create_result.content

                if chunk["choices"][0]["finish_reason"]:
                    create_result.content = content
                    final_chunk = create_result.model_dump()
                    ChatLogger(DEFAULT_OLLAMA_LOG_DIR, method=method).log_interaction(
                        messages,
                        final_chunk,
                        model=self._model_name,
                        tools=tools,
                        last_chunk=chunk,
                    )
            else:
                create_result = CreateResult(
                    content=str(chunk),
                    finish_reason="stop",
                    usage=RequestUsage(),
                    model=self._model_name,
                )
            yield create_result

    def _convert_mlx_to_create_result(self, response: ChatCompletionResponse) -> CreateResult:
        """Convert MLX ChatCompletionResponse to autogen CreateResult."""
        choice = response["choices"][0] if response["choices"] else {}
        message = choice.get("message", {})
        content = message.get("content", "") or ""
        finish_reason = choice.get("finish_reason", None)
        # Map invalid or None finish_reason to 'unknown'
        valid_finish_reasons = {'stop', 'length',
                                'function_calls', 'content_filter', 'unknown'}
        finish_reason = finish_reason if finish_reason in valid_finish_reasons else 'unknown'
        tool_calls = message.get("tool_calls", None)
        function_calls = [FunctionCall(
            id=f"call_{i}",
            name=call["function"]["name"],
            arguments=call["function"]["arguments"]
        ) for i, call in enumerate(tool_calls or [])]
        return CreateResult(
            content=content,
            finish_reason=finish_reason,
            usage=RequestUsage(
                prompt_tokens=response.get(
                    "usage", {}).get("prompt_tokens", 0),
                completion_tokens=response.get(
                    "usage", {}).get("completion_tokens", 0),
            ),
            model=response["model"],
            function_calls=function_calls if function_calls else None,
            cached=False,  # Set cached to False as response is not cached
        )


def convert_ollama_to_mlx(messages: Sequence[LLMMessage]) -> List[Dict[str, Any]]:
    """Serialize a sequence of autogen LLMMessages to a list of MLX Messages."""
    serialized = []
    for message in messages:
        if isinstance(message, UserMessage):
            content = message.content if isinstance(message.content, str) else "".join(
                part.text for part in message.content if hasattr(part, 'type') and part.type == 'text'
            )
            serialized.append({
                "role": "user",
                "content": content or None,
                "tool_calls": None
            })
        elif isinstance(message, SystemMessage):
            content = message.content if isinstance(message.content, str) else "".join(
                part.text for part in message.content if hasattr(part, 'type') and part.type == 'text'
            )
            serialized.append({
                "role": "system",
                "content": content or None,
                "tool_calls": None
            })
        elif isinstance(message, AssistantMessage):
            content = message.content if isinstance(message.content, str) else "".join(
                part.text for part in message.content if hasattr(part, 'type') and part.type == 'text'
            )
            tool_calls = convert_ollama_to_mlx_tool_calls(
                [message.function_call] if message.function_call else None
            ) if hasattr(message, "function_call") else []
            serialized.append({
                "role": "assistant",
                "content": content or None,
                "tool_calls": tool_calls
            })
        elif isinstance(message, FunctionExecutionResultMessage):
            serialized.append({
                "role": "tool",
                "content": str(message.content),
                "tool_calls": None
            })
        else:
            raise ValueError(f"Unknown message type: {type(message)}")
    return serialized


def convert_ollama_to_mlx_tool_calls(function_calls: Optional[List[FunctionCall]]) -> List[MLXToolCall]:
    """Convert autogen FunctionCalls to MLX ToolCalls."""
    if not function_calls:
        return []
    return [
        {
            "function": {
                "name": call.name,
                "arguments": call.arguments
            },
            "type": "function"
        } for call in function_calls
    ]
