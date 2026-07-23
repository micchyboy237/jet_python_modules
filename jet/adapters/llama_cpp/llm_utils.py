import json
import os
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from jet.adapters.llama_cpp.llm import ChatMessage, ToolCall
from jet.adapters.llama_cpp.types import LLAMACPP_LLM_TYPES
from jet.adapters.llama_cpp.utils import resolve_model_key
from jet.llm.config import DEFAULT_LOG_DIR
from jet.llm.logger_utils import ChatLogger
from jet.logger import CustomLogger
from jet.utils.text import format_sub_dir
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel


@dataclass
class LLMContext:
    """Holds clients, model name, logging, and verbosity."""

    model: str
    sync_client: OpenAI
    async_client: AsyncOpenAI
    verbose: bool = True
    logger: CustomLogger = field(default_factory=CustomLogger)
    chat_logger: ChatLogger | None = None

    @classmethod
    def from_params(
        cls,
        model: LLAMACPP_LLM_TYPES | None = None,
        base_url: str | None = None,
        api_key: str = "sk-1234",
        max_retries: int = 3,
        verbose: bool = True,
        agent_name: str | None = None,
        log_dir: str | None = None,
        logger: CustomLogger | None = None,
    ) -> "LLMContext":
        """Factory that mirrors LlamacppLLM.__init__ behaviour."""
        resolved_model = resolve_model_key(model or os.getenv("LLAMA_CPP_LLM_MODEL"))
        sync_client = OpenAI(
            base_url=base_url or os.getenv("LLAMA_CPP_LLM_URL"),
            api_key=api_key,
            max_retries=max_retries,
        )
        async_client = AsyncOpenAI(
            base_url=base_url or os.getenv("LLAMA_CPP_LLM_URL"),
            api_key=api_key,
            max_retries=max_retries,
        )
        _log_dir = log_dir or DEFAULT_LOG_DIR
        if agent_name:
            _log_dir = os.path.join(_log_dir, format_sub_dir(agent_name))
        return cls(
            model=resolved_model,
            sync_client=sync_client,
            async_client=async_client,
            verbose=verbose,
            logger=logger or CustomLogger(),
            chat_logger=ChatLogger(_log_dir),
        )


_DEFAULT_CTX: LLMContext | None = None


def get_default_context() -> LLMContext:
    """Return a cached default LLMContext, creating it on first access."""
    global _DEFAULT_CTX
    if _DEFAULT_CTX is None:
        _DEFAULT_CTX = LLMContext.from_params()
    return _DEFAULT_CTX


def set_default_context(ctx: LLMContext) -> None:
    """Override the module-level default context."""
    global _DEFAULT_CTX
    _DEFAULT_CTX = ctx


def reset_default_context() -> None:
    """Reset the default context (next access will rebuild from env)."""
    global _DEFAULT_CTX
    _DEFAULT_CTX = None


def _build_generation_params(
    top_k: int | None = None,
    enable_thinking: bool = False,
    seed: int | None = None,
) -> dict[str, Any]:
    """Build the extra_body dict for llama.cpp-specific params."""
    params: dict[str, Any] = {}
    if top_k is not None:
        params["top_k"] = top_k
    if seed is not None:
        params["seed"] = seed
    params["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    return params


def _build_create_kwargs(
    *,
    model: str,
    messages: list[ChatMessage] | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    top_p: float | None = None,
    presence_penalty: float | None = None,
    seed: int | None = None,
    stream: bool = False,
    stop: list[str] | None = None,
    top_k: int | None = None,
    enable_thinking: bool = False,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str = "auto",
    extra_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble kwargs dict for OpenAI chat.completions.create."""
    params = _build_generation_params(top_k, enable_thinking, seed)
    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "stream": stream,
        "extra_body": params,
    }
    if messages is not None:
        kwargs["messages"] = messages
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if top_p is not None:
        kwargs["top_p"] = top_p
    if presence_penalty is not None:
        kwargs["presence_penalty"] = presence_penalty
    if stop is not None:
        kwargs["stop"] = stop
    if tools is not None:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice
    if stream:
        kwargs["stream_options"] = {"include_usage": True}
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    return kwargs


def chat(
    messages: list[ChatMessage],
    temperature: float = 0.0,
    max_tokens: int | None = None,
    top_p: float | None = None,
    presence_penalty: float | None = None,
    top_k: int | None = None,
    seed: int | None = None,
    stream: bool = False,
    stop: list[str] | None = None,
    enable_thinking: bool = False,
    *,
    ctx: LLMContext | None = None,
) -> str | Iterator[str]:
    """Generate chat response (non-streaming or streaming)."""
    if ctx is None:
        ctx = get_default_context()
    kwargs = _build_create_kwargs(
        model=ctx.model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        presence_penalty=presence_penalty,
        seed=seed,
        stream=stream,
        stop=stop,
        top_k=top_k,
        enable_thinking=enable_thinking,
    )
    response = ctx.sync_client.chat.completions.create(**kwargs)
    if stream:

        def stream_generator() -> Iterator[str]:
            response_text = ""
            for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    content: str = delta.reasoning_content
                    if ctx.verbose:
                        ctx.logger.orange(content, flush=True, end="")
                    yield content
                    response_text += content
                elif hasattr(delta, "content") and delta.content:
                    content: str = delta.content
                    if ctx.verbose:
                        ctx.logger.teal(content, flush=True, end="")
                    yield content
                    response_text += content
            ctx.chat_logger.log_interaction(
                messages=messages,
                response=response_text,
                model=ctx.model,
                method="stream_chat",
            )

        return stream_generator()
    content = response.choices[0].message.content
    if ctx.verbose:
        ctx.logger.teal(content)
    ctx.chat_logger.log_interaction(
        messages=messages,
        response=content,
        model=ctx.model,
        method="chat",
    )
    return content


def generate(
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    seed: int | None = None,
    stream: bool = False,
    *,
    ctx: LLMContext | None = None,
) -> str | Iterator[str]:
    """Generate text completion from prompt."""
    if ctx is None:
        ctx = get_default_context()
    params = _build_generation_params(seed=seed)
    response = ctx.sync_client.completions.create(
        model=ctx.model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
        extra_body=params,
    )
    if stream:

        def stream_generator() -> Iterator[str]:
            response_text = ""
            for chunk in response:
                if chunk.choices and chunk.choices[0].text is not None:
                    content = chunk.choices[0].text
                    if ctx.verbose:
                        ctx.logger.teal(content, flush=True)
                    yield content
                    response_text += content
            ctx.chat_logger.log_interaction(
                messages=prompt,
                response=response_text,
                model=ctx.model,
                method="stream_generate",
            )

        return stream_generator()
    content = response.choices[0].text
    if ctx.verbose:
        ctx.logger.teal(content)
    ctx.chat_logger.log_interaction(
        messages=prompt,
        response=content,
        model=ctx.model,
        method="generate",
    )
    return content


def chat_with_tools(
    messages: list[ChatMessage],
    tools: list[dict[str, Any]],
    available_functions: dict[str, Callable[..., Any]],
    temperature: float = 0.0,
    max_tokens: int | None = None,
    top_p: float | None = None,
    presence_penalty: float | None = None,
    top_k: int | None = None,
    seed: int | None = None,
    stream: bool = False,
    enable_thinking: bool = False,
    *,
    ctx: LLMContext | None = None,
    **kwargs: Any,
) -> str | Iterator[str]:
    """
    Execute tool-calling loop with optional streaming.
    When stream=True, yields partial updates including tool calls and final response.
    """
    if ctx is None:
        ctx = get_default_context()
    tool_choice = kwargs.pop("tool_choice", "auto")
    create_kwargs = _build_create_kwargs(
        model=ctx.model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        presence_penalty=presence_penalty,
        seed=seed,
        stream=stream,
        top_k=top_k,
        enable_thinking=enable_thinking,
        tools=tools,
        tool_choice=tool_choice,
        extra_kwargs=kwargs,
    )
    if not stream:
        response = ctx.sync_client.chat.completions.create(**create_kwargs)
        message = response.choices[0].message
        tool_calls: list[ToolCall] = getattr(message, "tool_calls", []) or []
        if not tool_calls:
            content = message.content or ""
            if ctx.verbose:
                ctx.logger.teal(content)
            ctx.chat_logger.log_interaction(
                **{**create_kwargs, "response": content, "method": "chat"}
            )
            return content
        updated_messages: list[ChatMessage] = list(messages)
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
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ]
        updated_messages.append(assistant_msg)
        for tool_call in tool_calls:
            func_name = tool_call.function.name
            if ctx.verbose:
                ctx.logger.info(f"[TOOL EXEC] {func_name}")
            func = available_functions.get(func_name)
            if not func:
                ctx.logger.warning(f"Tool '{func_name}' not found. Skipping.")
                continue
            args = json.loads(tool_call.function.arguments)
            result = func(**args)
            if ctx.verbose:
                ctx.logger.debug(f"[TOOL OUT] {result}")
            updated_messages.append(
                {
                    "role": "tool",
                    "content": str(result),
                    "tool_call_id": tool_call.id,
                }
            )
        final_kwargs = {
            k: v
            for k, v in create_kwargs.items()
            if k not in ("messages", "tools", "tool_choice")
        }
        final_kwargs["messages"] = updated_messages
        final_kwargs["stream"] = False
        final_response = ctx.sync_client.chat.completions.create(**final_kwargs)
        final_content = final_response.choices[0].message.content
        if ctx.verbose:
            ctx.logger.teal(final_content)
        ctx.chat_logger.log_interaction(
            **{**create_kwargs, "response": final_content, "method": "chat"}
        )
        return final_content

    def stream_generator() -> Iterator[str]:
        response_text = ""
        updated_messages: list[ChatMessage] = list(messages)
        response = ctx.sync_client.chat.completions.create(**create_kwargs)
        message_content = ""
        tool_calls: list[ToolCall] = []
        for chunk in response:
            if not chunk.choices or not chunk.choices[0].delta:
                continue
            delta = chunk.choices[0].delta
            if delta.content is not None:
                message_content += delta.content
                response_text += delta.content
                if ctx.verbose:
                    ctx.logger.teal(delta.content, flush=True)
                yield delta.content
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    while len(tool_calls) <= idx:
                        tool_calls.append(
                            {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        )
                    tc = tool_calls[idx]
                    if tc_delta.id:
                        tc["id"] = tc_delta.id
                    if tc_delta.function and tc_delta.function.name:
                        tc["function"]["name"] += tc_delta.function.name
                    if tc_delta.function and tc_delta.function.arguments:
                        tc["function"]["arguments"] += tc_delta.function.arguments
        if tool_calls:
            assistant_msg: ChatMessage = {
                "role": "assistant",
                "content": message_content or None,
                "tool_calls": tool_calls,
            }
            updated_messages.append(assistant_msg)
            for tool_call in tool_calls:
                func_name = tool_call["function"]["name"]
                if ctx.verbose:
                    ctx.logger.info(f"[TOOL EXEC] {func_name}")
                func = available_functions.get(func_name)
                if not func:
                    ctx.logger.warning(f"Tool '{func_name}' not found. Skipping.")
                    continue
                args = json.loads(tool_call["function"]["arguments"])
                result = func(**args)
                if ctx.verbose:
                    ctx.logger.debug(f"[TOOL OUT] {result}")
                updated_messages.append(
                    {
                        "role": "tool",
                        "content": str(result),
                        "tool_call_id": tool_call["id"],
                    }
                )
        final_kwargs = {
            k: v
            for k, v in create_kwargs.items()
            if k not in ("messages", "tools", "tool_choice")
        }
        final_kwargs["messages"] = updated_messages
        final_kwargs["stream"] = True
        final_response = ctx.sync_client.chat.completions.create(**final_kwargs)
        final_content = ""
        for chunk in final_response:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                final_content += content
                if ctx.verbose:
                    ctx.logger.teal(content, flush=True)
                yield content
        ctx.chat_logger.log_interaction(
            **{**create_kwargs, "response": final_content, "method": "stream_chat"}
        )

    return stream_generator()


def chat_structured(
    messages: list[ChatMessage],
    response_model: type[BaseModel],
    temperature: float = 0.0,
    seed: int | None = None,
    *,
    ctx: LLMContext | None = None,
) -> BaseModel:
    """Generate structured JSON output using a Pydantic model."""
    if ctx is None:
        ctx = get_default_context()
    schema = response_model.model_json_schema()
    params = _build_generation_params(seed=seed)
    response = ctx.sync_client.chat.completions.create(
        model=ctx.model,
        messages=messages,
        response_format={"type": "json_object", "schema": schema},
        temperature=temperature,
        extra_body=params,
    )
    raw_json = response.choices[0].message.content or ""
    if ctx.verbose:
        ctx.logger.teal(raw_json)
    ctx.chat_logger.log_interaction(
        messages=messages,
        response=raw_json,
        model=ctx.model,
        response_format={"type": "json_object", "schema": schema},
        method="chat",
        temperature=temperature,
    )
    return response_model.model_validate_json(raw_json)


def chat_structured_stream(
    messages: list[ChatMessage],
    response_model: Any,
    temperature: float = 0.0,
    seed: int | None = None,
    *,
    ctx: LLMContext | None = None,
) -> Iterator[Any]:
    """
    Stream structured output with NO duplicates.
    - Single object → yields once
    - List[T] → yields only NEW items as they complete
    """
    if ctx is None:
        ctx = get_default_context()
    if hasattr(response_model, "model_json_schema"):
        schema = response_model.model_json_schema()
        validate_fn = response_model.model_validate_json
        is_list = False
    else:
        schema = response_model.json_schema()
        validate_fn = response_model.validate_json
        is_list = True
    params = _build_generation_params(seed=seed)
    response = ctx.sync_client.chat.completions.create(
        model=ctx.model,
        messages=messages,
        response_format={"type": "json_object", "schema": schema},
        temperature=temperature,
        stream=True,
        extra_body=params,
    )
    buffer = ""
    seen_items: list[Any] = []
    for chunk in response:
        if not chunk.choices or chunk.choices[0].delta.content is None:
            continue
        content: str = chunk.choices[0].delta.content
        if ctx.verbose:
            ctx.logger.teal(content, flush=True)
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
            new_items = parsed[len(seen_items) :]
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
                new_items = final_parsed[len(seen_items) :]
                for item in new_items:
                    seen_items.append(item)
                    yield item
        except Exception as e:
            if ctx.verbose:
                ctx.logger.warning(f"Final parse failed: {e}")
    ctx.chat_logger.log_interaction(
        messages=messages,
        response=seen_items if is_list else (seen_items[0] if seen_items else None),
        model=ctx.model,
        method="stream_chat",
        temperature=temperature,
        response_format={"type": "json_object", "schema": schema},
    )


async def achat(
    messages: list[ChatMessage],
    temperature: float = 0.0,
    max_tokens: int | None = None,
    top_p: float | None = None,
    presence_penalty: float | None = None,
    top_k: int | None = None,
    seed: int | None = None,
    stream: bool = False,
    stop: list[str] | None = None,
    enable_thinking: bool = False,
    *,
    ctx: LLMContext | None = None,
) -> str | AsyncIterator[str]:
    """Async chat completion (non-streaming or streaming)."""
    if ctx is None:
        ctx = get_default_context()
    params = _build_generation_params(top_k, enable_thinking, seed)
    response = await ctx.async_client.chat.completions.create(
        model=ctx.model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stream=stream,
        stop=stop,
        extra_body=params,
    )
    if stream:

        async def stream_generator() -> AsyncIterator[str]:
            response_text = ""
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    content: str = chunk.choices[0].delta.content
                    if ctx.verbose:
                        ctx.logger.teal(content, flush=True)
                    yield content
                    response_text += content
            ctx.chat_logger.log_interaction(
                messages=messages,
                response=response_text,
                model=ctx.model,
                method="stream_chat",
            )

        return stream_generator()
    content = response.choices[0].message.content
    if ctx.verbose:
        ctx.logger.teal(content)
    return content


async def agenerate(
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    seed: int | None = None,
    stream: bool = False,
    *,
    ctx: LLMContext | None = None,
) -> str | AsyncIterator[str]:
    """Async text completion (non-streaming or streaming)."""
    if ctx is None:
        ctx = get_default_context()
    params = _build_generation_params(seed=seed)
    response = await ctx.async_client.completions.create(
        model=ctx.model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
        extra_body=params,
    )
    if stream:

        async def stream_generator() -> AsyncIterator[str]:
            response_text = ""
            async for chunk in response:
                if chunk.choices and chunk.choices[0].text is not None:
                    content: str = chunk.choices[0].text
                    if ctx.verbose:
                        ctx.logger.teal(content, flush=True)
                    yield content
                    response_text += content
            ctx.chat_logger.log_interaction(
                messages=prompt,
                response=response_text,
                model=ctx.model,
                method="stream_generate",
            )

        return stream_generator()
    content = response.choices[0].text
    if ctx.verbose:
        ctx.logger.teal(content)
    return content


async def achat_with_tools(
    messages: list[ChatMessage],
    tools: list[dict[str, Any]],
    available_functions: dict[str, Callable[..., Any]],
    temperature: float = 0.0,
    max_tokens: int | None = None,
    top_p: float | None = None,
    presence_penalty: float | None = None,
    top_k: int | None = None,
    seed: int | None = None,
    stream: bool = False,
    enable_thinking: bool = False,
    *,
    ctx: LLMContext | None = None,
    **kwargs: Any,
) -> str | AsyncIterator[str]:
    """Async tool-calling loop with optional streaming."""
    if ctx is None:
        ctx = get_default_context()
    tool_choice = kwargs.pop("tool_choice", "auto")
    create_kwargs = _build_create_kwargs(
        model=ctx.model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        presence_penalty=presence_penalty,
        seed=seed,
        stream=stream,
        top_k=top_k,
        enable_thinking=enable_thinking,
        tools=tools,
        tool_choice=tool_choice,
        extra_kwargs=kwargs,
    )
    if not stream:
        response = await ctx.async_client.chat.completions.create(**create_kwargs)
        message = response.choices[0].message
        tool_calls: list[ToolCall] = getattr(message, "tool_calls", []) or []
        if not tool_calls:
            content = message.content or ""
            if ctx.verbose:
                ctx.logger.teal(content)
            return content
        updated_messages = list(messages)
        assistant_msg: ChatMessage = {
            "role": "assistant",
            "content": message.content or "",
        }
        updated_messages.append(assistant_msg)
        for tool_call in tool_calls:
            func_name = tool_call.function.name
            if func := available_functions.get(func_name):
                args = json.loads(tool_call.function.arguments)
                result = func(**args)
                updated_messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps({"result": result}),
                        "tool_call_id": tool_call.id,
                    }
                )
        final_response = await ctx.async_client.chat.completions.create(
            model=ctx.model,
            messages=updated_messages,
            temperature=temperature,
        )
        final_content = final_response.choices[0].message.content or ""
        if ctx.verbose:
            ctx.logger.teal(final_content)
        return final_content

    async def stream_generator() -> AsyncIterator[str]:
        response_text = ""
        response = await ctx.async_client.chat.completions.create(**create_kwargs)
        message_content = ""
        tool_calls: list[ToolCall] = []
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta
                if delta.content:
                    message_content += delta.content
                    if ctx.verbose:
                        ctx.logger.teal(delta.content, flush=True)
                    yield delta.content
                    response_text += delta.content
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        while len(tool_calls) <= idx:
                            tool_calls.append(
                                {
                                    "id": "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            )
                        tc = tool_calls[idx]
                        if tc_delta.id:
                            tc["id"] = tc_delta.id
                        if tc_delta.function.name:
                            tc["function"]["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            tc["function"]["arguments"] += tc_delta.function.arguments
        if not tool_calls:
            ctx.chat_logger.log_interaction(
                messages=messages,
                response=response_text,
                model=ctx.model,
                method="stream_chat",
            )
            return
        assistant_msg: ChatMessage = {
            "role": "assistant",
            "content": message_content,
            "tool_calls": tool_calls,
        }
        updated_messages = list(messages)
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
            yield f"[TOOL RESULT] {result_str}\n"
            response_text += f"[TOOL RESULT] {result_str}\n"
            updated_messages.append(
                {
                    "role": "tool",
                    "content": result_str,
                    "tool_call_id": tool_call["id"],
                }
            )
        final_create_kwargs = {
            k: v
            for k, v in create_kwargs.items()
            if k not in ("messages", "tools", "tool_choice")
        }
        final_create_kwargs["messages"] = updated_messages
        final_create_kwargs["stream"] = True
        final_response = await ctx.async_client.chat.completions.create(
            **final_create_kwargs
        )
        final_content = ""
        async for chunk in final_response:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                final_content += content
                response_text += content
                if ctx.verbose:
                    ctx.logger.teal(content, flush=True)
                yield content
        ctx.chat_logger.log_interaction(
            messages=messages,
            response=response_text,
            model=ctx.model,
            method="stream_chat",
        )

    return stream_generator()


async def achat_structured(
    messages: list[ChatMessage],
    response_model: type[BaseModel],
    temperature: float = 0.0,
    seed: int | None = None,
    *,
    ctx: LLMContext | None = None,
) -> BaseModel:
    """Async structured JSON output using Pydantic model."""
    if ctx is None:
        ctx = get_default_context()
    schema = response_model.model_json_schema()
    params = _build_generation_params(seed=seed)
    response = await ctx.async_client.chat.completions.create(
        model=ctx.model,
        messages=messages,
        response_format={"type": "json_object", "schema": schema},
        temperature=temperature,
        extra_body=params,
    )
    raw_json = response.choices[0].message.content or ""
    if ctx.verbose:
        ctx.logger.teal(raw_json)
    ctx.chat_logger.log_interaction(
        messages=messages,
        response=raw_json,
        model=ctx.model,
        response_format={"type": "json_object", "schema": schema},
        method="chat",
        temperature=temperature,
    )
    return response_model.model_validate_json(raw_json)


async def achat_structured_stream(
    messages: list[ChatMessage],
    response_model: Any,
    temperature: float = 0.0,
    seed: int | None = None,
    *,
    ctx: LLMContext | None = None,
) -> AsyncIterator[Any]:
    """
    Async structured streaming output with NO duplicates.
    - Single object → yields once when complete
    - List[T] → yields only NEW items as they become valid
    """
    if ctx is None:
        ctx = get_default_context()
    if hasattr(response_model, "model_json_schema"):
        schema = response_model.model_json_schema()
        validate_fn = response_model.model_validate_json
        is_list = False
    else:
        schema = response_model.json_schema()
        validate_fn = response_model.validate_json
        is_list = True
    params = _build_generation_params(seed=seed)
    response = await ctx.async_client.chat.completions.create(
        model=ctx.model,
        messages=messages,
        response_format={"type": "json_object", "schema": schema},
        temperature=temperature,
        stream=True,
        extra_body=params,
    )
    buffer = ""
    seen_items: list[Any] = []
    try:
        async for chunk in response:
            if not chunk.choices or chunk.choices[0].delta.content is None:
                continue
            content: str = chunk.choices[0].delta.content
            if ctx.verbose:
                ctx.logger.teal(content, flush=True)
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
                new_items = parsed[len(seen_items) :]
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
                    new_items = final_parsed[len(seen_items) :]
                    for item in new_items:
                        seen_items.append(item)
                        yield item
            except Exception as e:
                if ctx.verbose:
                    ctx.logger.warning(f"Final parse failed: {e}")
    finally:
        ctx.chat_logger.log_interaction(
            messages=messages,
            response=seen_items if is_list else (seen_items[0] if seen_items else None),
            model=ctx.model,
            method="stream_chat",
            temperature=temperature,
            response_format={"type": "json_object", "schema": schema},
        )


if __name__ == "__main__":
    import argparse

    from jet.adapters.llama_cpp.llm_utils import ChatMessage, chat

    parser = argparse.ArgumentParser(
        description="Stream chat completion using standalone llm_utils functions"
    )
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",
        default="Write a 2 sentence short story about a curious robot.",
        help="User input prompt for the chat model (default: %(default)s)",
    )
    parser.add_argument(
        "-s",
        "--system",
        type=str,
        default="You are a helpful assistant.",
        help="Optional system prompt for the chat model",
    )
    args = parser.parse_args()
    messages: list[ChatMessage] = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.prompt},
    ]
    print(f"System: {args.system}")
    print(f"User: {args.prompt}")
    print("Assistant: ", end="", flush=True)
    stream_response = chat(messages=messages, stream=True)
    for token in stream_response:
        pass
    print()
