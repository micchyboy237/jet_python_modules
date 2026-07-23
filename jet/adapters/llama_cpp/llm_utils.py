"""LLM utility functions for chat completions using llama.cpp OpenAI-compatible API."""

import argparse
import os
from typing import Dict, Iterable, List, Literal, Optional, Union

from jet.adapters.llama_cpp.factory import get_llm_client
from jet.adapters.llama_cpp.llm_types import (
    CompletionCreateParams,
    ModerationParam,
    PromptCacheOptionsParam,
    ResponseFormatParam,
    WebSearchOptionsParam,
)
from jet.libs.llama_cpp.utils.performance_tracker import PerformanceTracker, log_metrics
from jet.logger import logger
from openai import Stream
from openai.types.chat import (
    ChatCompletionAudioParam,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionPredictionContentParam,
    ChatCompletionStreamOptionsParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolUnionParam,
)
from openai.types.shared.reasoning_effort import ReasoningEffort
from openai.types.shared_params.metadata import Metadata

# ---------------------------------------------------------------------------
# Defaults (matching the original create_kwargs)
# ---------------------------------------------------------------------------

DEFAULT_MODEL = os.getenv("LLAMA_CPP_LLM_MODEL", "not-needed")
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_PRESENCE_PENALTY = 1.5
DEFAULT_STREAM = True
DEFAULT_STREAM_OPTIONS: ChatCompletionStreamOptionsParam = {"include_usage": True}
DEFAULT_EXTRA_BODY_TEMPLATE: Dict[str, object] = {
    "chat_template_kwargs": {
        "enable_thinking": False,
    },
}

# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

client = get_llm_client()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _deep_merge(base: Dict[str, object], override: Dict[str, object]) -> None:
    """Recursively merge override into base dict in-place."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)  # type: ignore[arg-type]
        else:
            base[key] = value


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chat_stream(
    user_prompt: str,
    system_prompt: str | None = None,
    enable_thinking: bool = False,
    verbose: bool = False,
    # ── Audio ───────────────────────────────────────────────────────────
    audio: Optional[ChatCompletionAudioParam] = None,
    # ── Sampling / token control ────────────────────────────────────────
    frequency_penalty: Optional[float] = None,
    logit_bias: Optional[Dict[str, int]] = None,
    logprobs: Optional[bool] = None,
    max_completion_tokens: Optional[int] = None,
    max_tokens: Optional[int] = DEFAULT_MAX_TOKENS,
    n: Optional[int] = None,
    presence_penalty: Optional[float] = DEFAULT_PRESENCE_PENALTY,
    seed: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    temperature: Optional[float] = DEFAULT_TEMPERATURE,
    top_logprobs: Optional[int] = None,
    top_p: Optional[float] = DEFAULT_TOP_P,
    # ── Streaming ───────────────────────────────────────────────────────
    stream: Optional[bool] = DEFAULT_STREAM,
    stream_options: Optional[ChatCompletionStreamOptionsParam] = DEFAULT_STREAM_OPTIONS,
    # ── Tools / function calling ────────────────────────────────────────
    tools: Optional[Iterable[ChatCompletionToolUnionParam]] = None,
    tool_choice: Optional[ChatCompletionToolChoiceOptionParam] = None,
    parallel_tool_calls: Optional[bool] = None,
    # ── Response format ─────────────────────────────────────────────────
    response_format: Optional[ResponseFormatParam] = None,
    # ── Reasoning ───────────────────────────────────────────────────────
    reasoning_effort: Optional[ReasoningEffort] = None,
    verbosity: Optional[Literal["low", "medium", "high"]] = None,
    # ── Modalities ──────────────────────────────────────────────────────
    modalities: Optional[List[Literal["text", "audio"]]] = None,
    # ── Moderation ──────────────────────────────────────────────────────
    moderation: Optional[ModerationParam] = None,
    # ── Predicted outputs ───────────────────────────────────────────────
    prediction: Optional[ChatCompletionPredictionContentParam] = None,
    # ── Prompt caching ──────────────────────────────────────────────────
    prompt_cache_key: Optional[str] = None,
    prompt_cache_options: Optional[PromptCacheOptionsParam] = None,
    prompt_cache_retention: Optional[Literal["in_memory", "24h"]] = None,
    # ── Metadata / identifiers ──────────────────────────────────────────
    metadata: Optional[Metadata] = None,
    safety_identifier: Optional[str] = None,
    user: Optional[str] = None,
    store: Optional[bool] = None,
    # ── Service tier ────────────────────────────────────────────────────
    service_tier: Optional[
        Literal["auto", "default", "flex", "scale", "priority"]
    ] = None,
    # ── Web search ──────────────────────────────────────────────────────
    web_search_options: Optional[WebSearchOptionsParam] = None,
    # ── Extra / transport ───────────────────────────────────────────────
    extra_headers: Optional[Dict[str, str]] = None,
    extra_query: Optional[Dict[str, object]] = None,
    extra_body: Optional[Dict[str, object]] = None,
    timeout: Optional[float] = None,
) -> str:
    """Stream a chat completion and return the full response text.

    Args:
        user_prompt: The user's input message.
        system_prompt: Optional system-level instruction.
        enable_thinking: If True, passes enable_thinking=True in chat_template_kwargs
                         so the model emits reasoning tokens (llama.cpp specific).
        verbose: If True, logs prompts and streams tokens to console with colors.

        audio: Parameters for audio output. Required when modalities includes "audio".
        frequency_penalty: Number between -2.0 and 2.0. Positive values penalize
                           repeated tokens.
        logit_bias: Modify likelihood of specified tokens. Maps token ID to bias (-100 to 100).
        logprobs: Whether to return log probabilities of output tokens.
        max_completion_tokens: Upper bound for generated tokens (including reasoning tokens).
        max_tokens: Maximum tokens to generate (deprecated, prefer max_completion_tokens).
        n: How many chat completion choices to generate.
        presence_penalty: Number between -2.0 and 2.0. Positive values encourage new topics.
        seed: Beta feature for deterministic sampling.
        stop: Up to 4 sequences where the API stops generating.
        temperature: Sampling temperature between 0 and 2.
        top_logprobs: Number of most likely tokens to return (0-20). Requires logprobs=True.
        top_p: Nucleus sampling parameter. Alternative to temperature.

        stream: If True, streams the response (default True).
        stream_options: Options for streaming response (e.g., {"include_usage": True}).

        tools: List of tools the model may call.
        tool_choice: Controls which tool is called (none/auto/required or specific tool).
        parallel_tool_calls: Whether to enable parallel function calling.

        response_format: Output format specification (text/json_object/json_schema).

        reasoning_effort: Constrains reasoning effort (none/minimal/low/medium/high/xhigh/max).
        verbosity: Constrains verbosity of response (low/medium/high).

        modalities: Output types to generate (text, audio).
        moderation: Configuration for running moderation on request and output.

        prediction: Static predicted output content for latency optimization.
        prompt_cache_key: Used to optimize cache hit rates.
        prompt_cache_options: Options for prompt caching (mode, ttl).
        prompt_cache_retention: Deprecated. Use prompt_cache_options.ttl instead.

        metadata: Up to 16 key-value pairs attached to the object.
        safety_identifier: Stable identifier for safety detection (max 64 chars).
        user: Deprecated. Use prompt_cache_key or safety_identifier.
        store: Whether to store the output for distillation/evals.

        service_tier: Processing type (auto/default/flex/scale/priority).
        web_search_options: Configuration for web search tool.

        extra_headers: Additional HTTP headers.
        extra_query: Additional query parameters.
        extra_body: Additional JSON body properties. Deep-merged over the template.
        timeout: Request timeout in seconds.

    Returns:
        The concatenated content string from the streamed response.
    """
    messages: List[ChatCompletionMessageParam] = []

    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": system_prompt,
            }
        )
        if verbose:
            logger.log("System prompt: ", system_prompt, colors=["PURPLE", "DEBUG"])

    messages.append(
        {
            "role": "user",
            "content": user_prompt,
        }
    )
    if verbose:
        logger.log("User prompt: ", user_prompt, colors=["GRAY", "DEBUG"])

    tracker = PerformanceTracker()

    # Start from template, deep-merge caller's extra_body, then set enable_thinking
    merged_extra_body: Dict[str, object] = dict(DEFAULT_EXTRA_BODY_TEMPLATE)
    if extra_body:
        _deep_merge(merged_extra_body, extra_body)
    if isinstance(merged_extra_body.get("chat_template_kwargs"), dict):
        merged_extra_body["chat_template_kwargs"]["enable_thinking"] = enable_thinking

    create_kwargs: CompletionCreateParams = {
        "model": DEFAULT_MODEL,
        "messages": messages,
        "audio": audio,
        "frequency_penalty": frequency_penalty,
        "logit_bias": logit_bias,
        "logprobs": logprobs,
        "max_completion_tokens": max_completion_tokens,
        "max_tokens": max_tokens,
        "n": n,
        "presence_penalty": presence_penalty,
        "seed": seed,
        "stop": stop,
        "temperature": temperature,
        "top_logprobs": top_logprobs,
        "top_p": top_p,
        "stream": stream,
        "stream_options": stream_options,
        "tools": tools,
        "tool_choice": tool_choice,
        "parallel_tool_calls": parallel_tool_calls,
        "response_format": response_format,
        "reasoning_effort": reasoning_effort,
        "verbosity": verbosity,
        "modalities": modalities,
        "moderation": moderation,
        "prediction": prediction,
        "prompt_cache_key": prompt_cache_key,
        "prompt_cache_options": prompt_cache_options,
        "prompt_cache_retention": prompt_cache_retention,
        "metadata": metadata,
        "safety_identifier": safety_identifier,
        "user": user,
        "store": store,
        "service_tier": service_tier,
        "web_search_options": web_search_options,
        "extra_headers": extra_headers,
        "extra_query": extra_query,
        "extra_body": merged_extra_body,
        "timeout": timeout,
    }

    # Remove None values so OpenAI client uses its own defaults
    create_kwargs = {k: v for k, v in create_kwargs.items() if v is not None}

    logger.debug(
        f"Calling chat.completions.create with keys: {list(create_kwargs.keys())}"
    )

    stream_response: Stream[ChatCompletionChunk] = client.chat.completions.create(
        **create_kwargs
    )

    content = ""
    for part in stream_response:
        if part.choices and part.choices[0].delta:
            delta = part.choices[0].delta
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                content += delta.reasoning_content
                tracker.mark_token()
                if verbose:
                    logger.orange(delta.reasoning_content, flush=True, end="")
            elif hasattr(delta, "content") and delta.content:
                content += delta.content
                tracker.mark_token()
                if verbose:
                    logger.teal(delta.content, flush=True, end="")

        usage = getattr(part, "usage", None)
        if usage is not None:
            metrics = tracker.finalize(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
            )
            logger.debug(f"Usage received: {metrics}")
            if verbose:
                log_metrics(metrics)

    logger.debug(f"Stream complete. Total content length: {len(content)} chars")
    return content


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stream chat completion from llama.cpp OpenAI API-compatible endpoint"
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

    user_prompt = args.prompt
    system_prompt = args.system
    verbose = True

    chat_stream(user_prompt, system_prompt, verbose=verbose)
