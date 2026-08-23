"""LLM Utilities Adapter for llama.cpp with Built-in Observability.

High-level sync/async interface to llama.cpp-compatible servers (OpenAI API
protocol), with automatic tracing (OpenTelemetry + Phoenix), agentic tool-use
loops, vision input, streaming, and flexible structured output.

QUICK REFERENCE — which function to call
==========================================
| Need                          | Sync         | Async          |
|--------------------------------|--------------|----------------|
| Chat / multi-turn / tools/vision | chat()     | achat()        |
| Raw text completion (no chat fmt)| generate() | agenerate()    |

Rule of thumb: use chat()/achat() unless you specifically need raw-prompt
completion with no chat template applied. generate()/agenerate() do NOT
support tools, vision, or structured output (response_format).

QUICK START
===========
    from jet.adapters.llama_cpp.llm_utils import chat

    result = chat("Explain OpenTelemetry in one sentence.")
    print(result.content)

Tool use (agentic loop, auto-executed):
    result = chat(
        "What's the weather in Tokyo?",
        tools=[{"type": "function", "function": {"name": "get_weather", ...}}],
        tool_registry={"get_weather": get_weather_fn},
    )

Structured output (Pydantic):
    class Answer(BaseModel):
        summary: str

    result = chat("Summarize X", response_format=Answer)
    if result.structured.success:
        answer = result.structured.parsed  # Answer instance

Vision:
    result = chat("What's in this image?", image_source="path/or/url.jpg")

STRUCTURED OUTPUT (response_format) — chat()/achat() ONLY
============================================================
Passed through structured_output.resolve_response_format(), which accepts:

| Input type                                   | Behavior                                      |
|-----------------------------------------------|------------------------------------------------|
| Pydantic BaseModel class                      | Schema injected, output validated & parsed into the model |
| JSON Schema dict (object or array)             | Wrapped as json_schema format; needs "properties"/"$schema" or "type":"array"+"items" |
| Grammar dict {"type":"grammar","grammar":"<GBNF>"} | Routed to extra_body (not response_format) per llama.cpp; guarantees valid output at token level |
| Raw dict {"type":"json_object"/"json_schema",...} | Passed through unchanged |
| None (default)                                | Plain text, no parsing |

⚠️ CONSTRAINT: grammar output requires enable_thinking=False.
⚠️ generate()/agenerate() cannot use response_format at all.

Result access: after streaming, result.structured is a StructuredResult with
.success (bool), .parsed (typed instance or dict), .error, .validation_errors.
Always check result.structured.success before reading .parsed.

PARAMETERS BY GROUP
====================
(Not every function accepts every param — see each function's signature.)

Input:
    prompt_or_messages (str | list[dict]): prompt string or OpenAI messages list.
        Default: "What is OpenTelemetry in one sentence?"
    model (str): model id served by llama.cpp. Default: MODEL env var or
        "qwen3.5-uncensored:2b".
    image_source (str | None): local path, URL, or bytes for vision input.
        chat()/achat() only.

Sampling:
    max_tokens (int, default 16384), temperature (float, default 0.7,
        use 0.0–0.3 for reliable structured output), top_p (0.8), top_k (20,
        llama.cpp native), min_p (0.0, disabled), repeat_penalty (1.1),
        presence_penalty (1.5, range -2..2), frequency_penalty (0.0, range -2..2),
        logit_bias (dict[str,int] | None, e.g. {"1234": -100}), seed (int | None),
        stop (list[str] | None, max 4).
    enable_thinking (bool, default False): request reasoning tokens.
        MUST be False when response_format uses grammar.

Tools (chat()/achat() only):
    tools (list[dict] | None): OpenAI-format function definitions.
    tool_choice (str | dict | None): "auto" | "none" | "required" | specific fn.
    tool_registry (dict[str, Callable] | None): name→executor map. Providing
        this enables an automatic agentic loop (model calls tool → executor
        runs → result fed back → repeat). Omit it to get raw tool_calls back
        for you to handle yourself.
    max_tool_rounds (int, default 10): cap on agentic loop iterations.
        Only matters when tool_registry is set.

Structured output (chat()/achat() only):
    response_format (Any): see STRUCTURED OUTPUT section above.

Observability:
    project_name (str): Phoenix project for trace grouping. "" disables
        observability. Default: "<func>-llm-utils-obs".
    capture_content (bool, default True): record prompt/response text in
        traces. Set False for PII-sensitive workloads.
    phoenix_url (str): Phoenix server base URL. Default: PHOENIX_URL const.
    session_id (str | None): groups related traces as one conversation thread.

Client / misc:
    client (OpenAI | AsyncOpenAI | None): pre-configured client; if None,
        one is created via get_llm_client()/get_async_llm_client().
    extra_body_params (dict | None): merged into the API request's extra_body
        for llama.cpp-specific params. Grammar is auto-routed here when
        response_format resolves to grammar type.

FUNCTIONS
=========
    chat(prompt_or_messages, ...) -> StreamCompletionResult
        Sync multi-turn chat. Supports tools, vision, structured output.
    achat(prompt_or_messages, ...) -> StreamCompletionResult
        Async version of chat().
    generate(prompt, ...) -> StreamCompletionResult
        Sync raw completion, no chat formatting, no tools/vision/structured output.
    agenerate(prompt, ...) -> StreamCompletionResult
        Async version of generate().

NOTES
=====
- Observability auto-configures whenever project_name is non-empty.
- Structured-output parsing happens after the stream completes and attaches
  to result.structured — always check .success before using .parsed.
- All four functions return StreamCompletionResult (see chat_stream_types.py):
  .content (str), .tool_calls (list), .usage (dict|None), .finish_reason (str|None),
  .has_tool_calls (bool property), .structured (only set when response_format used).
"""

import argparse
from typing import Any, Callable

from jet.libs.llama_cpp.usage.chat_stream_observability import (
    MODEL,
    PHOENIX_URL,
    run_chat_stream,
    run_generate_stream,
)
from jet.libs.llama_cpp.usage.chat_stream_observability_async import (
    run_chat_stream_async,
    run_generate_stream_async,
)
from jet.libs.llama_cpp.usage.chat_stream_types import (
    StreamCompletionResult,
)
from openai import AsyncOpenAI, OpenAI


def chat(
    prompt_or_messages: str
    | list[dict[str, Any]] = "What is OpenTelemetry in one sentence?",
    model: str = MODEL,
    *,
    project_name: str = "chat-llm-utils-obs",
    capture_content: bool = True,
    phoenix_url: str = PHOENIX_URL,
    image_source: str | None = None,
    client: OpenAI | None = None,
    enable_thinking: bool = False,
    max_tokens: int = 16384,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    min_p: float = 0.0,
    repeat_penalty: float = 1.1,
    presence_penalty: float = 1.5,
    frequency_penalty: float = 0.0,
    logit_bias: dict[str, int] | None = None,
    seed: int | None = None,
    stop: list[str] | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    tool_registry: dict[str, Callable[..., Any]] | None = None,
    response_format: Any = None,
    max_tool_rounds: int = 10,
    extra_body_params: dict[str, Any] | None = None,
    session_id: str | None = None,
) -> StreamCompletionResult:
    """Synchronous multi-turn chat with optional tool execution and structured output."""
    from jet.logger import logger

    logger.debug(f"💬 chat() called with type={type(prompt_or_messages).__name__}")
    return run_chat_stream(
        prompt_or_messages=prompt_or_messages,
        model=model,
        project_name=project_name,
        capture_content=capture_content,
        phoenix_url=phoenix_url,
        image_source=image_source,
        client=client,
        enable_thinking=enable_thinking,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repeat_penalty=repeat_penalty,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        logit_bias=logit_bias,
        seed=seed,
        stop=stop,
        tools=tools,
        tool_choice=tool_choice,
        tool_registry=tool_registry,
        response_format=response_format,
        max_tool_rounds=max_tool_rounds,
        extra_body_params=extra_body_params,
        session_id=session_id,
    )


async def achat(
    prompt_or_messages: str
    | list[dict[str, Any]] = "What is OpenTelemetry in one sentence?",
    model: str = MODEL,
    *,
    project_name: str = "achat-llm-utils-obs",
    capture_content: bool = True,
    phoenix_url: str = PHOENIX_URL,
    image_source: str | None = None,
    client: AsyncOpenAI | None = None,
    enable_thinking: bool = False,
    max_tokens: int = 16384,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    min_p: float = 0.0,
    repeat_penalty: float = 1.1,
    presence_penalty: float = 1.5,
    frequency_penalty: float = 0.0,
    logit_bias: dict[str, int] | None = None,
    seed: int | None = None,
    stop: list[str] | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    tool_registry: dict[str, Callable[..., Any]] | None = None,
    response_format: Any = None,
    max_tool_rounds: int = 10,
    extra_body_params: dict[str, Any] | None = None,
    session_id: str | None = None,
) -> StreamCompletionResult:
    """Async multi-turn chat with optional tool execution and structured output."""
    from jet.logger import logger

    logger.debug(f"💬 achat() called with type={type(prompt_or_messages).__name__}")
    return await run_chat_stream_async(
        prompt_or_messages=prompt_or_messages,
        model=model,
        project_name=project_name,
        capture_content=capture_content,
        phoenix_url=phoenix_url,
        image_source=image_source,
        client=client,
        enable_thinking=enable_thinking,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repeat_penalty=repeat_penalty,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        logit_bias=logit_bias,
        seed=seed,
        stop=stop,
        tools=tools,
        tool_choice=tool_choice,
        tool_registry=tool_registry,
        response_format=response_format,
        max_tool_rounds=max_tool_rounds,
        extra_body_params=extra_body_params,
        session_id=session_id,
    )


def generate(
    prompt: str,
    model: str = MODEL,
    *,
    project_name: str = "generate-llm-utils-obs",
    capture_content: bool = True,
    phoenix_url: str = PHOENIX_URL,
    client: OpenAI | None = None,
    max_tokens: int = 16384,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    min_p: float = 0.0,
    repeat_penalty: float = 1.1,
    presence_penalty: float = 1.5,
    frequency_penalty: float = 0.0,
    logit_bias: dict[str, int] | None = None,
    seed: int | None = None,
    stop: list[str] | None = None,
    extra_body_params: dict[str, Any] | None = None,
    session_id: str | None = None,
) -> StreamCompletionResult:
    """Synchronous raw text generation alternative to chat()."""
    from jet.logger import logger

    logger.debug(f"✏️ generate() called with prompt length={len(prompt)}")
    return run_generate_stream(
        prompt=prompt,
        model=model,
        project_name=project_name,
        capture_content=capture_content,
        phoenix_url=phoenix_url,
        client=client,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repeat_penalty=repeat_penalty,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        logit_bias=logit_bias,
        seed=seed,
        stop=stop,
        extra_body_params=extra_body_params,
        session_id=session_id,
    )


async def agenerate(
    prompt: str,
    model: str = MODEL,
    *,
    project_name: str = "agenerate-llm-utils-obs",
    capture_content: bool = True,
    phoenix_url: str = PHOENIX_URL,
    client: AsyncOpenAI | None = None,
    max_tokens: int = 16384,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    min_p: float = 0.0,
    repeat_penalty: float = 1.1,
    presence_penalty: float = 1.5,
    frequency_penalty: float = 0.0,
    logit_bias: dict[str, int] | None = None,
    seed: int | None = None,
    stop: list[str] | None = None,
    extra_body_params: dict[str, Any] | None = None,
    session_id: str | None = None,
) -> StreamCompletionResult:
    """Asynchronous raw text generation alternative to achat()."""
    from jet.logger import logger

    logger.debug(f"✏️ agenerate() called with prompt length={len(prompt)}")
    return await run_generate_stream_async(
        prompt=prompt,
        model=model,
        project_name=project_name,
        capture_content=capture_content,
        phoenix_url=phoenix_url,
        client=client,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repeat_penalty=repeat_penalty,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        logit_bias=logit_bias,
        seed=seed,
        stop=stop,
        extra_body_params=extra_body_params,
        session_id=session_id,
    )


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream vision-model chat completions with Phoenix observability."
    )
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",
        default="What is OpenTelemetry in one sentence?",
        help="Prompt for the chat/image analysis.",
    )
    parser.add_argument(
        "-i",
        "--image-source",
        type=str,
        default=None,
        help="Path or URL to an image to analyze. Omit for a text-only chat request.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL,
        help="Model name to request (env: LLAMA_CPP_VISION_MODEL).",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="achat-llm-utils-obs",
        help="Phoenix project name to log traces under.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible generation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    import asyncio

    from jet.logger import logger

    args = get_args()
    result = asyncio.run(
        achat(
            args.prompt,
            model=args.model,
            image_source=args.image_source,
            project_name=args.project,
            seed=args.seed,
        )
    )
    logger.info(
        f"📋 Result: {len(result.content)} chars, finish_reason={result.finish_reason}"
    )
