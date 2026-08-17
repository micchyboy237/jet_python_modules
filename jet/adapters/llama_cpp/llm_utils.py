"""LLM Utilities Adapter for llama.cpp with Built-in Observability.

This module provides a simplified, high-level interface for interacting with
llama.cpp-compatible servers via the OpenAI API protocol. It serves as the
primary entry point for LLM operations within the jet framework.

Key Features:
    - Unified sync/async API for chat completions and raw text generation
    - Automatic OpenTelemetry tracing and Phoenix observability integration
    - Native support for agentic tool-use loops with executable registries
    - Vision/multimodal support via base64 image encoding
    - Streaming responses with real-time console output and usage metrics
    - Dynamic structured output: pass Pydantic models, JSON Schema dicts,
      grammar (GBNF), or raw response_format dicts — all auto-resolved

Structured Output Support:
    The `response_format` parameter accepts multiple input types that are
    automatically normalized by structured_output.resolve_response_format():

      1. Pydantic BaseModel class
         → Generates JSON Schema, injects schema prompt, validates output
         → Access result via: result.structured.parsed (typed model instance)

      2. JSON Schema dict (object or array)
         → Wraps in json_schema API format, injects field descriptions
         → Must have "properties"/"$schema" (object) or "type":"array"+"items"

      3. Grammar dict {"type": "grammar", "grammar": "<GBNF>"}
         → Routes grammar to extra_body (not response_format) per llama.cpp spec
         → Guarantees valid output at token level; requires enable_thinking=False

      4. Raw dict {"type": "json_object"} or {"type": "json_schema", ...}
         → Passed through directly to the API

      5. None (default)
         → Plain text mode, no structured parsing

    After streaming completes, output is automatically parsed and validated.
    Results attach to StreamCompletionResult.structured (a StructuredResult
    dataclass with .success, .parsed, .error, .validation_errors fields).

Common Args:
    These parameters are shared across chat(), achat(), generate(), and agenerate().
    Not all functions accept every parameter; check each function's signature.

    prompt_or_messages (str | list[dict]): Input prompt string or pre-formatted
        OpenAI messages list. Defaults to "What is OpenTelemetry in one sentence?".
    model (str): Model identifier served by the llama.cpp backend.
        Defaults to MODEL env var or "qwen3.5-uncensored:2b".
    project_name (str): Phoenix project name for trace grouping. Set to "" to
        disable observability setup. Defaults to "<func>-llm-utils-obs".
    capture_content (bool): Whether to record prompt/response text in traces.
        Set False for PII-sensitive workloads. Defaults to True.
    phoenix_url (str): Base URL of the Phoenix observability server.
        Defaults to PHOENIX_URL constant.
    image_source (str | None): Local path, HTTP(S) URL, or raw bytes of an image
        to include as vision input. Only applies to chat/achat. Defaults to None.
    client (OpenAI | AsyncOpenAI | None): Pre-configured client instance.
        If None, a default client is created via get_client()/get_async_client().
    enable_thinking (bool): Request model reasoning/thinking tokens in output.
        Only supported by models with native thinking capability. MUST be False
        when using grammar-constrained output. Defaults to False.
    max_tokens (int): Maximum completion tokens to generate. Defaults to 16384.
    temperature (float): Sampling temperature (0.0 = greedy, 2.0 = most random).
        Use 0.0–0.3 for structured output reliability. Defaults to 0.7.
    top_p (float): Nucleus sampling threshold. Tokens with cumulative probability
        above this are excluded. Defaults to 0.8.
    top_k (int): Limit sampling to the k most likely tokens. llama.cpp native
        parameter passed via extra_body. Defaults to 20.
    min_p (float): Minimum probability threshold relative to top token.
        llama.cpp native parameter. Defaults to 0.0 (disabled).
    repeat_penalty (float): Repetition penalty factor. 1.0 = no penalty.
        llama.cpp native parameter. Defaults to 1.1.
    presence_penalty (float): Penalize tokens based on presence in context so far.
        Range: -2.0 to 2.0. Defaults to 1.5.
    frequency_penalty (float): Penalize tokens proportional to their frequency.
        Range: -2.0 to 2.0. Defaults to 0.0.
    logit_bias (dict[str, int] | None): Token ID → bias mapping to steer
        generation. E.g. {"1234": -100} suppresses token 1234. Defaults to None.
    seed (int | None): Random seed for reproducible outputs. None = random.
    stop (list[str] | None): Up to 4 stop sequences that halt generation.
    tools (list[dict] | None): OpenAI-format tool/function definitions.
        Only applies to chat/achat. Defaults to None.
    tool_choice (str | dict | None): Tool selection strategy: "auto", "none",
        "required", or a specific function dict. Defaults to None.
    tool_registry (dict[str, Callable] | None): Mapping of tool names to callable
        executors. When provided, enables automatic agentic tool-call loops.
        Without it, raw tool calls are returned for external handling.
    response_format (Any): Structured output format. Accepts Pydantic BaseModel
        class, JSON Schema dict, grammar dict, raw API dict, or None. See
        "Structured Output Support" section above for full details. Defaults to None.
    max_tool_rounds (int): Maximum agentic loop iterations before forced stop.
        Only effective when tool_registry is provided. Defaults to 10.
    extra_body_params (dict | None): Additional key-value pairs merged into the
        API request's extra_body field for llama.cpp-specific parameters.
        Grammar is automatically routed here when using grammar response_format.
    session_id (str | None): Session identifier to group related traces as a
        conversation thread in Phoenix. Defaults to None.

Functions:
    chat(): Synchronous multi-turn chat with optional tool execution and
            structured output. Returns StreamCompletionResult.
    achat(): Async multi-turn chat with optional tool execution and
             structured output. Returns StreamCompletionResult.
    generate(): Synchronous raw text completion (no chat formatting).
                Does not support structured output. Returns StreamCompletionResult.
    agenerate(): Async raw text completion (no chat formatting).
                 Does not support structured output. Returns StreamCompletionResult.

Note:
    All functions automatically configure tracing when project_name is non-empty.
    Structured output parsing happens post-stream and attaches to
    result.structured. Check result.structured.success before accessing
    result.structured.parsed. See individual function signatures for which
    Common Args each accepts.
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
