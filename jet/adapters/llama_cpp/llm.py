import json
import os
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Union, Literal, TypedDict
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel
from jet.adapters.llama_cpp.utils import resolve_model_value
from jet.llm.config import DEFAULT_LOG_DIR
from jet.llm.logger_utils import ChatLogger
from jet.logger import CustomLogger
from jet.utils.text import format_sub_dir


# === Strict TypedDict for OpenAI-compatible messages ===
class ToolFunction(TypedDict):
    name: str
    arguments: str


class ToolCall(TypedDict):
    id: str
    type: Literal["function"]
    function: ToolFunction

class ChatMessage(TypedDict, total=False):
    """OpenAI chat message with optional tool_call_id only for role='tool'."""
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_call_id: str  # Required only when role == "tool"
    tool_calls: List[ToolCall]

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
        verbose: bool = True,
        agent_name: Optional[str] = None,
        log_dir: str = DEFAULT_LOG_DIR,
        logger: Optional[CustomLogger] = None,
    ):
        """Initialize sync and async clients with model resolution."""
        self.model = resolve_model_value(model)
        self.sync_client = OpenAI(base_url=base_url, api_key=api_key, max_retries=max_retries)
        self.async_client = AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=max_retries)
        self.verbose = verbose

        if agent_name:
            log_dir = os.path.join(log_dir, format_sub_dir(agent_name))

        self._chat_logger = ChatLogger(log_dir)
        self._logger = logger or CustomLogger()

    # === Sync Chat ===
    def chat(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.0,
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
                            self._logger.teal(content, flush=True)
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
            self._logger.teal(content)

        self._chat_logger.log_interaction(
            messages=messages,
            response=content,
            model=self.model,
            method="chat",
        )

        return content

    # === Sync Completions ===
    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
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
                            self._logger.teal(content, flush=True)
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
            self._logger.teal(content)

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
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[str, Iterator[str]]:
        """
        Execute tool-calling loop with optional streaming.
        When stream=True, yields partial updates including tool calls and final response.
        """
        tool_choice = kwargs.get("tool_choice") or "auto"
        create_kwargs = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "temperature": temperature,
            "tool_choice": tool_choice,
            "stream": stream,
            **({ "max_tokens": max_tokens } if max_tokens is not None else {}),
            **{k: v for k, v in kwargs.items() if k != "tool_choice"},
        }

        if not stream:
            # Existing non-streaming path
            response = self.sync_client.chat.completions.create(**create_kwargs)
            message = response.choices[0].message
            tool_calls: List[ToolCall] = getattr(message, "tool_calls", []) or []
            if not tool_calls:
                content = message.content or ""
                if self.verbose:
                    self._logger.teal(content)
                self._chat_logger.log_interaction(**{
                    **create_kwargs,
                    "response": content,
                    "method": "chat",
                })
                return content

            updated_messages: List[ChatMessage] = messages.copy()
            assistant_msg: ChatMessage = {
                "role": "assistant",
                "content": message.content or "",
            }
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

            for tool_call in tool_calls:
                func_name = tool_call.function.name
                if self.verbose:
                    self._logger.info(f"[TOOL EXEC] {func_name}")
                func = available_functions.get(func_name)
                if not func:
                    self._logger.warning(f"Tool '{func_name}' not found. Skipping.")
                    continue

                args = json.loads(tool_call.function.arguments)
                result = func(**args)

                if self.verbose:
                    self._logger.debug(f"[TOOL OUT] {result}")

                tool_response: ChatMessage = {
                    "role": "tool",
                    "content": str(result),
                    "tool_call_id": tool_call.id,
                }
                updated_messages.append(tool_response)

            final_kwargs = {
                k: v for k, v in create_kwargs.items()
                if k not in ["messages", "tools", "tool_choice"]
            }
            final_kwargs["messages"] = updated_messages
            final_kwargs["stream"] = False

            final_response = self.sync_client.chat.completions.create(**final_kwargs)
            final_content = final_response.choices[0].message.content
            if self.verbose:
                self._logger.teal(final_content)
            self._chat_logger.log_interaction(**{
                **create_kwargs,
                "response": final_content,
                "method": "chat",
            })
            return final_content

        # === STREAMING PATH ===
        def stream_generator() -> Iterator[str]:
            response_text = ""
            updated_messages: List[ChatMessage] = messages.copy()

            # First call: detect tool calls (stream content + tool_calls)
            response = self.sync_client.chat.completions.create(**create_kwargs)
            message_content = ""
            tool_calls: List[ToolCall] = []

            for chunk in response:
                if not chunk.choices or not chunk.choices[0].delta:
                    continue
                delta = chunk.choices[0].delta

                # Stream assistant content
                if delta.content is not None:
                    message_content += delta.content
                    response_text += delta.content
                    if self.verbose:
                        self._logger.teal(delta.content, flush=True)
                    yield delta.content

                # Accumulate tool calls
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        while len(tool_calls) <= idx:
                            tool_calls.append({
                                "id": "", "type": "function",
                                "function": {"name": "", "arguments": ""}
                            })
                        tc = tool_calls[idx]
                        if tc_delta.id:
                            tc["id"] = tc_delta.id
                        if tc_delta.function and tc_delta.function.name:
                            tc["function"]["name"] += tc_delta.function.name
                        if tc_delta.function and tc_delta.function.arguments:
                            tc["function"]["arguments"] += tc_delta.function.arguments

            # === EXECUTE TOOLS (NO YIELD) ===
            if tool_calls:
                assistant_msg: ChatMessage = {
                    "role": "assistant",
                    "content": message_content or None,
                    "tool_calls": tool_calls
                }
                updated_messages.append(assistant_msg)

                for tool_call in tool_calls:
                    func_name = tool_call["function"]["name"]
                    if self.verbose:
                        self._logger.info(f"[TOOL EXEC] {func_name}")

                    func = available_functions.get(func_name)
                    if not func:
                        self._logger.warning(f"Tool '{func_name}' not found. Skipping.")
                        continue
                    else:
                        args = json.loads(tool_call["function"]["arguments"])
                        result = func(**args)

                    if self.verbose:
                        self._logger.debug(f"[TOOL OUT] {result}")

                    updated_messages.append({
                        "role": "tool",
                        "content": str(result),
                        "tool_call_id": tool_call["id"],
                    })

            # === FINAL LLM CALL (STREAM ONLY CONTENT) ===
            final_kwargs = {
                k: v for k, v in create_kwargs.items()
                if k not in ["messages", "tools", "tool_choice"]
            }
            final_kwargs["messages"] = updated_messages
            final_kwargs["stream"] = True

            final_response = self.sync_client.chat.completions.create(**final_kwargs)
            final_content = ""
            for chunk in final_response:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    final_content += content
                    if self.verbose:
                        self._logger.teal(content, flush=True)
                    yield content

            # === FINAL LOG ===
            self._chat_logger.log_interaction(**{
                **create_kwargs,
                "response": final_content,
                "method": "stream_chat",
            })

        return stream_generator()

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
            self._logger.teal(raw_json)

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
                self._logger.teal(content, flush=True)
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
                    self._logger.warning(f"Final parse failed: {e}")

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
        temperature: float = 0.0,
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
                            self._logger.teal(content, flush=True)
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
            self._logger.teal(content)
        return content

    # === Async Completions ===
    async def agenerate(
        self,
        prompt: str,
        temperature: float = 0.0,
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
                            self._logger.teal(content, flush=True)
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
            self._logger.teal(content)
        return content

    # === Async Tools ===
    async def achat_with_tools(
        self,
        messages: List[ChatMessage],
        tools: List[Dict[str, Any]],
        available_functions: Dict[str, Callable[..., Any]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[str, AsyncIterator[str]]:
        tool_choice = kwargs.get("tool_choice") or "auto"
        create_kwargs = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "temperature": temperature,
            "tool_choice": tool_choice,
            "stream": stream,
            **({ "max_tokens": max_tokens } if max_tokens is not None else {}),
            **{k: v for k, v in kwargs.items() if k != "tool_choice"},
        }

        if not stream:
            # Existing non-stream path (unchanged)
            response = await self.async_client.chat.completions.create(**create_kwargs)
            message = response.choices[0].message
            tool_calls: List[ToolCall] = getattr(message, "tool_calls", []) or []
            if not tool_calls:
                content = message.content or ""
                if self.verbose:
                    self._logger.teal(content)
                return content

            updated_messages = messages.copy()
            assistant_msg: ChatMessage = {"role": "assistant", "content": message.content or ""}
            updated_messages.append(assistant_msg)

            for tool_call in tool_calls:
                func_name = tool_call.function.name
                if func := available_functions.get(func_name):
                    args = json.loads(tool_call.function.arguments)
                    result = func(**args)
                    updated_messages.append({
                        "role": "tool",
                        "content": json.dumps({"result": result}),
                        "tool_call_id": tool_call.id,
                    })

            final_response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=updated_messages,
                temperature=temperature,
            )
            final_content = final_response.choices[0].message.content or ""
            if self.verbose:
                self._logger.teal(final_content)
            return final_content

        # === ASYNC STREAMING PATH ===
        async def stream_generator() -> AsyncIterator[str]:
            response_text = ""
            tool_results = []

            response = await self.async_client.chat.completions.create(**create_kwargs)
            message_content = ""
            tool_calls: List[ToolCall] = []
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        message_content += delta.content
                        if self.verbose:
                            self._logger.teal(delta.content, flush=True)
                        yield delta.content
                        response_text += delta.content
                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            while len(tool_calls) <= idx:
                                tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                            tc = tool_calls[idx]
                            if tc_delta.id:
                                tc["id"] = tc_delta.id
                            if tc_delta.function.name:
                                tc["function"]["name"] += tc_delta.function.name
                            if tc_delta.function.arguments:
                                tc["function"]["arguments"] += tc_delta.function.arguments

            if not tool_calls:
                self._chat_logger.log_interaction(
                    messages=messages,
                    response=response_text,
                    model=self.model,
                    method="stream_chat",
                )
                return

            assistant_msg: ChatMessage = {
                "role": "assistant",
                "content": message_content,
                "tool_calls": tool_calls,
            }
            updated_messages = messages.copy()
            updated_messages.append(assistant_msg)

            for tool_call in tool_calls:
                func_name = tool_call["function"]["name"]
                yield f"\n[TOOL CALL] {func_name}\n"
                response_text += f"\n[TOOL CALL] {func_name}\n"
                func = available_functions.get(func_name)
                if not func:
                    result_str = json.dumps({"error": f"Tool {func_name} not found"})
                else:
                    try:
                        args = json.loads(tool_call["function"]["arguments"])
                        result = func(**args)
                        result_str = json.dumps({"result": result}, ensure_ascii=False)
                    except Exception as e:
                        result_str = json.dumps({"error": str(e)})
                tool_results.append(result_str)
                yield f"[TOOL RESULT] {result_str}\n"
                response_text += f"[TOOL RESULT] {result_str}\n"
                updated_messages.append({
                    "role": "tool",
                    "content": result_str,
                    "tool_call_id": tool_call["id"],
                })

            final_create_kwargs = {
                k: v for k, v in create_kwargs.items() if k not in ["messages", "tools", "tool_choice"]
            }
            final_create_kwargs["messages"] = updated_messages
            final_create_kwargs["stream"] = True

            final_response = await self.async_client.chat.completions.create(**final_create_kwargs)
            final_content = ""
            async for chunk in final_response:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    final_content += content
                    response_text += content
                    if self.verbose:
                        self._logger.teal(content, flush=True)
                    yield content

            self._chat_logger.log_interaction(
                messages=messages,
                response=response_text,
                model=self.model,
                method="stream_chat",
            )

        return stream_generator()

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
            self._logger.teal(raw_json)

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
                    self._logger.teal(content, flush=True)
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
                        self._logger.warning(f"Final parse failed: {e}")
        finally:
            self._chat_logger.log_interaction(
                messages=messages,
                response=seen_items if is_list else (seen_items[0] if seen_items else None),
                model=self.model,
                method="stream_chat",
                temperature=temperature,
                response_format={"type": "json_object", "schema": schema},
            )
