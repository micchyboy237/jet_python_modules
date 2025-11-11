import json
import os
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Union, Literal, TypedDict
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel
from jet.adapters.llama_cpp.utils import resolve_model_value
from jet.llm.config import DEFAULT_LOG_DIR
from jet.llm.logger_utils import ChatLogger
from jet.logger import logger
from jet.utils.text import format_sub_dir


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
        base_url: str = "http://shawn-pc.local:8080/v1",
        api_key: str = "sk-1234",
        max_retries: int = 3,
        verbose: bool = False,
        agent_name: Optional[str] = None,
        log_dir: str = DEFAULT_LOG_DIR,
    ):
        """Initialize sync and async clients with model resolution."""
        self.model = resolve_model_value(model)
        self.sync_client = OpenAI(base_url=base_url, api_key=api_key, max_retries=max_retries)
        self.async_client = AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=max_retries)
        self.verbose = verbose

        if agent_name:
            log_dir = os.path.join(log_dir, format_sub_dir(agent_name))

        self._chat_logger = ChatLogger(log_dir)

    # === Sync Chat ===
    def chat(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        stop: Optional[List[str]] = None,
    ) -> Union[str, Iterator[str]]:
        """Generate chat response (non-streaming or streaming)."""
        response = self.sync_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            stop=stop,
        )
        if stream:
            def stream_generator() -> Iterator[str]:
                response_text = ""
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content is not None:
                        content: str = chunk.choices[0].delta.content
                        if self.verbose:
                            logger.teal(content, flush=True)
                        yield content
                        response_text += content

                self._chat_logger.log_interaction(
                    messages=messages,
                    response=response_text,
                    model=self.model,
                    method="stream_chat",
                )

            return stream_generator()

        content = response.choices[0].message.content
        if self.verbose:
            logger.teal(content)

        self._chat_logger.log_interaction(
            messages=messages,
            response=content,
            model=self.model,
            method="chat",
        )

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
        response_text = ""
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                content: str = chunk.choices[0].delta.content
                if self.verbose:
                    logger.teal(content, flush=True)
                yield chunk
                response_text += content

        self._chat_logger.log_interaction(
            messages=messages,
            response=response_text,
            model=self.model,
            method="stream_chat",
        )

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
        response_text = ""
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                content: str = chunk.choices[0].delta.content
                if self.verbose:
                    logger.teal(content, flush=True)
                yield chunk
                response_text += content

        self._chat_logger.log_interaction(
            messages=messages,
            response=response_text,
            model=self.model,
            method="stream_chat",
        )

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
                response_text = ""
                for chunk in response:
                    if chunk.choices and chunk.choices[0].text is not None:
                        content = chunk.choices[0].text
                        if self.verbose:
                            logger.teal(content, flush=True)
                        yield content
                        response_text += content

                self._chat_logger.log_interaction(
                    messages=prompt,
                    response=response_text,
                    model=self.model,
                    method="stream_generate",
                )

            return stream_generator()
        content = response.choices[0].text
        if self.verbose:
            logger.teal(content)

        self._chat_logger.log_interaction(
            messages=prompt,
            response=content,
            model=self.model,
            method="generate",
        )

        return content

    # === Tools (Sync) ===
    def chat_with_tools(
        self,
        messages: List[ChatMessage],
        tools: List[Dict[str, Any]],
        available_functions: Dict[str, Callable[..., Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,        # ← Control response length
        **kwargs: Any,                           # ← Accept any extra OpenAI params (e.g. stop, metadata)
    ) -> str:
        """
        Execute a complete tool-calling loop in one synchronous call:
        1. Send user messages + tools → LLM decides to call tools or respond
        2. If tool calls → execute via `available_functions`
        3. Append tool results as `role: tool` messages
        4. Final LLM call → generate natural language response

        Fully compatible with LangChain's `bind_tools` and `langgraph_bigtool`.

        Args:
            messages: Conversation history in OpenAI format
            tools: List of tool definitions (from `convert_to_openai_tool`)
            available_functions: Mapping of tool name → actual callable
            temperature: Sampling temperature
            max_tokens: Limit output length (optional)
            **kwargs: Extra args passed to OpenAI API (e.g. stop, presence_penalty)

        Returns:
            Final assistant response as string
        """
        # Build shared kwargs for both LLM calls
        tool_choice = kwargs.get("tool_choice") or "auto"
        create_kwargs = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "temperature": temperature,
            "tool_choice": tool_choice,
            **({ "max_tokens": max_tokens } if max_tokens is not None else {}),
            **{k: v for k, v in kwargs.items() if k != "tool_choice"},
        }

        # Step 1: Initial LLM call — may return tool_calls
        response = self.sync_client.chat.completions.create(**create_kwargs)
        message = response.choices[0].message
        tool_calls: List[ToolCall] = getattr(message, "tool_calls", []) or []

        # If no tool calls → return direct response
        if not tool_calls:
            content = message.content or ""
            if self.verbose:
                logger.teal(content)

            self._chat_logger.log_interaction(**{
                **create_kwargs,
                "response": content,
                "method": "chat",
            })

            return content

        # Step 2: Build updated message history with assistant intent
        updated_messages: List[ChatMessage] = messages.copy()
        assistant_msg: ChatMessage = {
            "role": "assistant",
            "content": message.content or "",
        }
        # Preserve structured tool_calls for LangChain parsing
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in tool_calls
            ]
        updated_messages.append(assistant_msg)

        # Step 3: Execute each tool call
        for tool_call in tool_calls:
            func_name = tool_call.function.name
            if self.verbose:
                logger.info(f"Executing tool: {func_name}")

            func = available_functions.get(func_name)
            if not func:
                logger.warning(f"Tool '{func_name}' not found in available_functions. Skipping.")
                continue

            try:
                # Parse arguments from JSON string
                args = json.loads(tool_call.function.arguments)
                if self.verbose:
                    logger.debug(f"Tool {func_name} input: {args}")

                # Execute tool
                result = func(**args)

                # Serialize result
                result_str = json.dumps({"result": result}, ensure_ascii=False)

                # Truncate long outputs in logs
                ellipsis = "..." if len(result_str) > 200 else ""
                if self.verbose:
                    logger.debug(f"Tool {func_name} output: {result_str[:200]}{ellipsis}")

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse arguments for {func_name}: {e}")
                result_str = json.dumps({"error": "Invalid JSON in tool arguments"})
            except Exception as e:
                logger.error(f"Tool {func_name} execution failed: {e}")
                result_str = json.dumps({"error": str(e)})

            # Append tool response to history
            tool_response: ChatMessage = {
                "role": "tool",
                "content": result_str,
                "tool_call_id": tool_call.id,
            }
            updated_messages.append(tool_response)

        # Step 4: Final LLM call with tool results
        final_response = self.sync_client.chat.completions.create(
            **{k: v for k, v in create_kwargs.items() if k != "messages"},
            messages=updated_messages
        )
        final_content = final_response.choices[0].message.content or ""
        if self.verbose:
            logger.teal(final_content)

        self._chat_logger.log_interaction(**{
            **create_kwargs,
            "response": final_content,
            "method": "chat",
        })

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

        self._chat_logger.log_interaction(**{
            "messages": messages,
            "response": raw_json,
            "model": self.model,
            "response_format": {"type": "json_object", "schema": response_model.model_json_schema()},
            "method": "chat",
            "temperature": temperature,
        })

        return response_model.model_validate_json(raw_json)

    # === Sync Structured Stream ===
    def chat_structured_stream(
        self,
        messages: List[ChatMessage],
        response_model: Any,  # BaseModel or TypeAdapter(List[T])
        temperature: float = 0.0,
    ) -> Iterator[Any]:
        """
        Stream structured output with NO duplicates.
        - Single object → yields once
        - List[T] → yields only NEW items as they complete
        """
        # Determine schema and validator
        if hasattr(response_model, "model_json_schema"):  # single BaseModel
            schema = response_model.model_json_schema()
            validate_fn = response_model.model_validate_json
            is_list = False
        else:  # TypeAdapter(List[...])
            schema = response_model.json_schema()
            validate_fn = response_model.validate_json
            is_list = True

        response = self.sync_client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object", "schema": schema},
            temperature=temperature,
            stream=True,
        )

        buffer = ""
        seen_items: list[Any] = []  # Track yielded items (for lists only)

        for chunk in response:
            if not chunk.choices or chunk.choices[0].delta.content is None:
                continue
            content: str = chunk.choices[0].delta.content
            if self.verbose:
                logger.teal(content, flush=True)
            buffer += content

            stripped = buffer.strip()
            if not (stripped.startswith("{") or stripped.startswith("[")):
                continue

            try:
                parsed = validate_fn(stripped)

                if not is_list:
                    # Single object: yield once and done
                    seen_items.append(parsed)
                    yield parsed
                    buffer = ""  # prevent re-yielding
                    continue

                # List mode: yield only NEW items
                new_items = parsed[len(seen_items):]
                for item in new_items:
                    seen_items.append(item)
                    yield item

            except Exception:
                # Not complete yet
                pass

        # Final parse
        if buffer.strip():
            try:
                final_parsed = validate_fn(buffer.strip())

                if not is_list:
                    if final_parsed not in seen_items:
                        seen_items.append(final_parsed)
                        yield final_parsed
                else:
                    new_items = final_parsed[len(seen_items):]
                    for item in new_items:
                        seen_items.append(item)
                        yield item

            except Exception as e:
                if self.verbose:
                    logger.warning(f"Final parse failed: {e}")

        self._chat_logger.log_interaction(
            messages=messages,
            response=seen_items if is_list else (seen_items[0] if seen_items else None),
            model=self.model,
            method="stream_chat",
            temperature=temperature,
            response_format={"type": "json_object", "schema": schema},
        )

    # === Async Chat ===
    async def achat(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        stop: Optional[List[str]] = None,
    ) -> Union[str, AsyncIterator[str]]:
        """Async chat completion (non-streaming or streaming)."""
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            stop=stop,
        )
        if stream:
            async def stream_generator() -> AsyncIterator[str]:
                response_text = ""
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content is not None:
                        content: str = chunk.choices[0].delta.content
                        if self.verbose:
                            logger.teal(content, flush=True)
                        yield content
                        response_text += content

                self._chat_logger.log_interaction(
                    messages=messages,
                    response=response_text,
                    model=self.model,
                    method="stream_chat",
                )

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
                response_text = ""
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].text is not None:
                        content: str = chunk.choices[0].text
                        if self.verbose:
                            logger.teal(content, flush=True)
                        yield content
                        response_text += content

                self._chat_logger.log_interaction(
                    messages=prompt,
                    response=response_text,
                    model=self.model,
                    method="stream_generate",
                )

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

        self._chat_logger.log_interaction(**{
            "messages": messages,
            "response": raw_json,
            "model": self.model,
            "response_format": {"type": "json_object", "schema": response_model.model_json_schema()},
            "method": "chat",
            "temperature": temperature,
        })

        return response_model.model_validate_json(raw_json)

    # === Async Structured Stream ===
    async def achat_structured_stream(
        self,
        messages: List[ChatMessage],
        response_model: Any,
        temperature: float = 0.0,
    ) -> AsyncIterator[Any]:
        """
        Async structured streaming output with NO duplicates.
        - Single object → yields once when complete
        - List[T] → yields only NEW items as they become valid
        """
        if hasattr(response_model, "model_json_schema"):
            schema = response_model.model_json_schema()
            validate_fn = response_model.model_validate_json
            is_list = False
        else:
            schema = response_model.json_schema()
            validate_fn = response_model.validate_json
            is_list = True
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object", "schema": schema},
            temperature=temperature,
            stream=True,
        )
        buffer = ""
        seen_items: list[Any] = []
        try:
            async for chunk in response:
                if not chunk.choices or chunk.choices[0].delta.content is None:
                    continue
                content: str = chunk.choices[0].delta.content
                if self.verbose:
                    logger.teal(content, flush=True)
                buffer += content
                stripped = buffer.strip()
                if not (stripped.startswith("{") or stripped.startswith("[")):
                    continue
                try:
                    parsed = validate_fn(stripped)
                    if not is_list:
                        seen_items.append(parsed)
                        yield parsed
                        buffer = ""
                        continue
                    new_items = parsed[len(seen_items):]
                    for item in new_items:
                        seen_items.append(item)
                        yield item
                except Exception:
                    pass
            if buffer.strip():
                try:
                    final_parsed = validate_fn(buffer.strip())
                    if not is_list:
                        if final_parsed not in seen_items:
                            seen_items.append(final_parsed)
                            yield final_parsed
                    else:
                        new_items = final_parsed[len(seen_items):]
                        for item in new_items:
                            seen_items.append(item)
                            yield item
                except Exception as e:
                    if self.verbose:
                        logger.warning(f"Final parse failed: {e}")
        finally:
            self._chat_logger.log_interaction(
                messages=messages,
                response=seen_items if is_list else (seen_items[0] if seen_items else None),
                model=self.model,
                method="stream_chat",
                temperature=temperature,
                response_format={"type": "json_object", "schema": schema},
            )
