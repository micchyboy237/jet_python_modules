import json
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Union, Literal, TypedDict
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel
from jet.adapters.llama_cpp.utils import resolve_model_value
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
        model: str = "qwen3-instruct-2507:4b",
        base_url: str = "http://localhost:8080/v1",
        api_key: str = "sk-1234",
        max_retries: int = 3,
        verbose: bool = False,
    ):
        """Initialize sync and async clients with model resolution."""
        self.model = resolve_model_value(model)
        self.sync_client = OpenAI(base_url=base_url, api_key=api_key, max_retries=max_retries)
        self.async_client = AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=max_retries)
        self.verbose = verbose

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
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )
        if stream:
            def stream_generator() -> Iterator[str]:
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content is not None:
                        content: str = chunk.choices[0].delta.content
                        if self.verbose:
                            logger.teal(content, flush=True)
                        yield content
            return stream_generator()

        content = response.choices[0].message.content
        if self.verbose:
            logger.teal(content)
        return content

    # === Sync Chat Stream ===
    def chat_stream(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Iterator[ChatCompletion]:
        response = self.sync_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                content: str = chunk.choices[0].delta.content
                if self.verbose:
                    logger.teal(content, flush=True)
                yield chunk

    # === Async Chat Stream ===
    async def achat_stream(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[ChatCompletion]:
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                content: str = chunk.choices[0].delta.content
                if self.verbose:
                    logger.teal(content, flush=True)
                yield chunk

    # === Sync Completions ===
    def generate(
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
            def stream_generator() -> Iterator[str]:
                for chunk in response:
                    if chunk.choices and chunk.choices[0].text is not None:
                        content = chunk.choices[0].text
                        if self.verbose:
                            logger.teal(content, flush=True)
                        yield content
            return stream_generator()
        content = response.choices[0].text
        if self.verbose:
            logger.teal(content)
        return content

    # === Tools (Sync) ===
    def chat_with_tools(
        self,
        messages: List[ChatMessage],
        tools: List[Dict[str, Any]],
        available_functions: Dict[str, Callable[..., Any]],
        temperature: float = 0.7,
    ) -> str:
        """Handle tool calling in a single sync call with final response and optional verbose logging."""
        response = self.sync_client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            tools=tools,
            tool_choice="auto",
            temperature=temperature,
        )
        message = response.choices[0].message
        tool_calls: List[ToolCall] = getattr(message, "tool_calls", []) or []

        # No tool calls → return assistant content directly
        if not tool_calls:
            content = message.content or ""
            if self.verbose:
                logger.teal(content)
            return content

        # Build updated message history
        updated_messages: List[ChatMessage] = messages.copy()
        assistant_msg: ChatMessage = {
            "role": "assistant",
            "content": message.content or "",
        }
        updated_messages.append(assistant_msg)  # type: ignore[arg-type]

        # Execute each requested tool
        for tool_call in tool_calls:
            func_name = tool_call.function.name
            if self.verbose:
                logger.info(f"Calling tool: {func_name}")
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

        # Final LLM call to synthesize the answer
        final_response = self.sync_client.chat.completions.create(
            model=self.model,
            messages=updated_messages,  # type: ignore[arg-type]
            temperature=temperature,
        )
        final_content = final_response.choices[0].message.content or ""
        if self.verbose:
            logger.teal(final_content)
        return final_content

    # === Structured Outputs (Sync) ===
    def chat_structured(
        self,
        messages: List[ChatMessage],
        response_model: type[BaseModel],
        temperature: float = 0.0,
    ) -> BaseModel:
        """Generate structured JSON output using a Pydantic model with optional verbose logging."""
        response = self.sync_client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            response_format={"type": "json_object", "schema": response_model.model_json_schema()},
            temperature=temperature,
        )
        raw_json = response.choices[0].message.content or ""
        if self.verbose:
            logger.teal(raw_json)
        return response_model.model_validate_json(raw_json)

    # === Async Chat ===
    async def achat(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> Union[str, AsyncIterator[str]]:
        """Async chat completion (non-streaming or streaming)."""
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )
        if stream:
            async def stream_generator() -> AsyncIterator[str]:
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content is not None:
                        content: str = chunk.choices[0].delta.content
                        if self.verbose:
                            logger.teal(content, flush=True)
                        yield content
            return stream_generator()

        content = response.choices[0].message.content
        if self.verbose:
            logger.teal(content)
        return content

    # === Async Completions ===
    async def agenerate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> Union[str, AsyncIterator[str]]:
        """Async text completion (non-streaming or streaming)."""
        response = await self.async_client.completions.create(
            model=self.model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )
        if stream:
            async def stream_generator() -> AsyncIterator[str]:
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].text is not None:
                        content: str = chunk.choices[0].text
                        if self.verbose:
                            logger.teal(content, flush=True)
                        yield content
            return stream_generator()

        content = response.choices[0].text
        if self.verbose:
            logger.teal(content)
        return content

    # === Async Tools ===
    async def achat_with_tools(
        ################################################################################
        # NOTE: The sync version already exists above; this replaces the incomplete stub #
        ################################################################################
        self,
        messages: List[ChatMessage],
        tools: List[Dict[str, Any]],
        available_functions: Dict[str, Callable[..., Any]],
        temperature: float = 0.7,
    ) -> str:
        """Async tool calling with final response and optional verbose logging."""
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
            content = message.content or ""
            if self.verbose:
                logger.teal(content)
            return content

        updated_messages: List[ChatMessage] = messages.copy()
        assistant_msg: ChatMessage = {
            "role": "assistant",
            "content": message.content or "",
        }
        updated_messages.append(assistant_msg)  # type: ignore[arg-type]

        for tool_call in tool_calls:
            func_name = tool_call.function.name
            if self.verbose:
                logger.info(f"Calling tool: {func_name}")
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
        final_content = final_response.choices[0].message.content or ""
        if self.verbose:
            logger.teal(final_content)
        return final_content

    # === Async Structured Outputs ===
    async def achat_structured(
        self,
        messages: List[ChatMessage],
        response_model: type[BaseModel],
        temperature: float = 0.0,
    ) -> BaseModel:
        """Async structured JSON output using Pydantic model with verbose logging."""
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            response_format={"type": "json_object", "schema": response_model.model_json_schema()},
            temperature=temperature,
        )
        raw_json = response.choices[0].message.content or ""
        if self.verbose:
            logger.teal(raw_json)
        return response_model.model_validate_json(raw_json)
