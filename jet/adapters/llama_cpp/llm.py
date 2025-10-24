import json
from typing import Any, Callable, Dict, Iterator, List, Optional, Union, Literal, TypedDict
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel
from jet.adapters.llama_cpp.models import resolve_model_value
from jet.logger import logger


# === Strict TypedDict for OpenAI-compatible messages ===
class ChatMessage(TypedDict, total=False):
    """OpenAI chat message with optional tool_call_id only for role='tool'."""
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_call_id: str  # Required only when role == "tool"


class ToolFunction(TypedDict):
    name: str
    arguments: str


class ToolCall(TypedDict):
    id: str
    type: Literal["function"]
    function: ToolFunction


class LlamacppLLM:
    """
    A client for LLM interactions via llama-server using OpenAI-compatible API.
    Supports sync/async chat, completions, streaming, tools, and structured outputs.
    """

    def __init__(
        self,
        model: str = "Qwen_Qwen3-4B-Instruct-2507-Q4_K_M",
        base_url: str = "http://localhost:8080/v1",
        api_key: str = "sk-1234",
        max_retries: int = 3,
    ):
        """Initialize sync and async clients with model resolution."""
        self.model = resolve_model_value(model)
        self.sync_client = OpenAI(base_url=base_url, api_key=api_key, max_retries=max_retries)
        self.async_client = AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=max_retries)

    # === Sync Chat ===
    def chat(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> Union[str, Iterator[str]]:
        """Generate chat response (non-streaming or streaming)."""
        response = self.sync_client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )
        if stream:
            return (chunk.choices[0].delta.content or "" for chunk in response if chunk.choices)
        return response.choices[0].message.content

    # === Sync Completions ===
    def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> Union[str, Iterator[str]]:
        """Generate text completion from prompt."""
        response = self.sync_client.completions.create(
            model=self.model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )
        if stream:
            return (chunk.choices[0].text or "" for chunk in response)
        return response.choices[0].text

    # === Tools (Sync) ===
    def chat_with_tools(
        self,
        messages: List[ChatMessage],
        tools: List[Dict[str, Any]],
        available_functions: Dict[str, Callable[..., Any]],
        temperature: float = 0.7,
    ) -> str:
        """Handle tool calling in a single sync call with final response."""
        response = self.sync_client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            tools=tools,
            tool_choice="auto",
            temperature=temperature,
        )
        message = response.choices[0].message
        tool_calls: List[ToolCall] = getattr(message, "tool_calls", []) or []

        if not tool_calls:
            return message.content or ""

        updated_messages: List[ChatMessage] = messages.copy()
        assistant_msg: ChatMessage = {
            "role": "assistant",
            "content": message.content or "",
        }
        # Add tool_calls to assistant message if present
        if tool_calls:
            # OpenAI client doesn't include tool_calls in model_dump(), so reconstruct
            assistant_msg["content"] = message.content or None  # type: ignore
            # We'll skip full serialization; rely on message history

        updated_messages.append(assistant_msg)  # type: ignore[arg-type]

        for tool_call in tool_calls:
            func_name = tool_call.function.name
            if func := available_functions.get(func_name):
                args = json.loads(tool_call.function.arguments)
                result = func(**args)
                tool_response: ChatMessage = {
                    "role": "tool",
                    "content": json.dumps({"result": result}),
                    "tool_call_id": tool_call.id,
                }
                updated_messages.append(tool_response)
            else:
                logger.warning(f"Function {func_name} not found")

        final_response = self.sync_client.chat.completions.create(
            model=self.model,
            messages=updated_messages,  # type: ignore[arg-type]
            temperature=temperature,
        )
        return final_response.choices[0].message.content

    # === Structured Outputs ===
    def chat_structured(
        self,
        messages: List[ChatMessage],
        response_model: type[BaseModel],
        temperature: float = 0.0,
    ) -> BaseModel:
        """Generate structured JSON output using Pydantic model."""
        response = self.sync_client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            response_format={"type": "json_object", "schema": response_model.model_json_schema()},
            temperature=temperature,
        )
        return response_model.model_validate_json(response.choices[0].message.content)

    # === Async Chat ===
    async def achat(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Async chat completion."""
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    # === Async Completions ===
    async def acomplete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Async text completion."""
        response = await self.async_client.completions.create(
            model=self.model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].text

    # === Async Tools ===
    async def achat_with_tools(
        self,
        messages: List[ChatMessage],
        tools: List[Dict[str, Any]],
        available_functions: Dict[str, Callable[..., Any]],
        temperature: float = 0.7,
    ) -> str:
        """Async tool calling with final response."""
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            tools=tools,
            tool_choice="auto",
            temperature=temperature,
        )
        message = response.choices[0].message
        tool_calls: List[ToolCall] = getattr(message, "tool_calls", []) or []

        if not tool_calls:
            return message.content or ""

        updated_messages: List[ChatMessage] = messages.copy()
        assistant_msg: ChatMessage = {
            "role": "assistant",
            "content": message.content or "",
        }
        updated_messages.append(assistant_msg)  # type: ignore[arg-type]

        for tool_call in tool_calls:
            func_name = tool_call.function.name
            if func := available_functions.get(func_name):
                args = json.loads(tool_call.function.arguments)
                result = func(**args)
                tool_response: ChatMessage = {
                    "role": "tool",
                    "content": json.dumps({"result": result}),
                    "tool_call_id": tool_call.id,
                }
                updated_messages.append(tool_response)
            else:
                logger.warning(f"Function {func_name} not found")

        final_response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=updated_messages,  # type: ignore[arg-type]
            temperature=temperature,
        )
        return final_response.choices[0].message.content