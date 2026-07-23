"""Type definitions for chat completion parameters.

Reuses OpenAI's existing types where available, defines custom TypedDicts
only for parameters without a public export from openai.types.chat.*.
"""

from typing import Dict, Iterable, List, Literal, Optional, TypedDict, Union

from openai.types.chat import (
    ChatCompletionAudioParam,
    ChatCompletionMessageParam,
    ChatCompletionPredictionContentParam,
    ChatCompletionStreamOptionsParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolUnionParam,
)
from openai.types.shared.reasoning_effort import ReasoningEffort
from openai.types.shared_params.metadata import Metadata

__all__ = [
    "ResponseFormatParam",
    "WebSearchOptionsParam",
    "PromptCacheOptionsParam",
    "ModerationParam",
    "FunctionCallParam",
    "FunctionParam",
    "CompletionCreateParams",
]


# ---------------------------------------------------------------------------
# Custom TypedDicts (not publicly exported by openai.types.chat.*)
# ---------------------------------------------------------------------------


class ResponseFormatParam(TypedDict, total=False):
    """An object specifying the format that the model must output.

    Setting to {"type": "json_schema", "json_schema": {...}} enables Structured Outputs.
    Setting to {"type": "json_object"} enables older JSON mode.
    """

    type: Literal["text", "json_object", "json_schema"]
    json_schema: Optional[Dict[str, object]]


class WebSearchOptionsParam(TypedDict, total=False):
    """Configuration for the web search tool."""

    search_context_size: Optional[Literal["low", "medium", "high"]]
    user_location: Optional[Dict[str, str]]


class PromptCacheOptionsParam(TypedDict, total=False):
    """Options for prompt caching. Supported for gpt-5.6 and later models."""

    mode: Optional[Literal["auto", "explicit"]]
    ttl: Optional[str]  # "30m" is currently the only supported value


class ModerationParam(TypedDict, total=False):
    """Configuration for running moderation on the request input and generated output."""

    model: str


# ── Deprecated params (from completion_create_params) ────────────────────


class FunctionCallParam(TypedDict, total=False):
    """Deprecated in favor of tool_choice. Controls which function is called by the model."""

    name: str


class FunctionParam(TypedDict, total=False):
    """Deprecated in favor of tools. A function the model may generate JSON inputs for."""

    name: str
    description: Optional[str]
    parameters: Dict[str, object]


# ---------------------------------------------------------------------------
# Main completion params
# ---------------------------------------------------------------------------


class CompletionCreateParams(TypedDict, total=False):
    """TypedDict covering all parameters accepted by chat.completions.create().

    Uses OpenAI's existing types where available, defines custom TypedDicts
    only for params without a public export.
    """

    # ── Required-ish (depends on overload) ──────────────────────────────
    messages: Iterable[ChatCompletionMessageParam]
    model: str

    # ── Audio ───────────────────────────────────────────────────────────
    audio: Optional[ChatCompletionAudioParam]

    # ── Sampling / token control ────────────────────────────────────────
    frequency_penalty: Optional[float]
    logit_bias: Optional[Dict[str, int]]
    logprobs: Optional[bool]
    max_completion_tokens: Optional[int]
    max_tokens: Optional[int]
    n: Optional[int]
    presence_penalty: Optional[float]
    seed: Optional[int]
    stop: Optional[Union[str, List[str]]]
    temperature: Optional[float]
    top_logprobs: Optional[int]
    top_p: Optional[float]

    # ── Streaming ───────────────────────────────────────────────────────
    stream: Optional[bool]
    stream_options: Optional[ChatCompletionStreamOptionsParam]

    # ── Tools / function calling ────────────────────────────────────────
    tools: Optional[Iterable[ChatCompletionToolUnionParam]]
    tool_choice: Optional[ChatCompletionToolChoiceOptionParam]
    parallel_tool_calls: Optional[bool]

    # ── Deprecated function calling ─────────────────────────────────────
    function_call: Optional[Union[Literal["none", "auto"], FunctionCallParam]]
    functions: Optional[Iterable[FunctionParam]]

    # ── Response format ─────────────────────────────────────────────────
    response_format: Optional[ResponseFormatParam]

    # ── Reasoning ───────────────────────────────────────────────────────
    reasoning_effort: Optional[ReasoningEffort]
    verbosity: Optional[Literal["low", "medium", "high"]]

    # ── Modalities ──────────────────────────────────────────────────────
    modalities: Optional[List[Literal["text", "audio"]]]

    # ── Moderation ──────────────────────────────────────────────────────
    moderation: Optional[ModerationParam]

    # ── Predicted outputs ───────────────────────────────────────────────
    prediction: Optional[ChatCompletionPredictionContentParam]

    # ── Prompt caching ──────────────────────────────────────────────────
    prompt_cache_key: Optional[str]
    prompt_cache_options: Optional[PromptCacheOptionsParam]
    prompt_cache_retention: Optional[Literal["in_memory", "24h"]]

    # ── Metadata / identifiers ──────────────────────────────────────────
    metadata: Optional[Metadata]
    safety_identifier: Optional[str]
    user: Optional[str]
    store: Optional[bool]

    # ── Service tier ────────────────────────────────────────────────────
    service_tier: Optional[Literal["auto", "default", "flex", "scale", "priority"]]

    # ── Web search ──────────────────────────────────────────────────────
    web_search_options: Optional[WebSearchOptionsParam]

    # ── Extra / transport ───────────────────────────────────────────────
    extra_headers: Optional[Dict[str, str]]
    extra_query: Optional[Dict[str, object]]
    extra_body: Optional[Dict[str, object]]
    timeout: Optional[float]
