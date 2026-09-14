import json
from contextlib import contextmanager
from typing import Any, Generator

from openinference.semconv.trace import (
    DocumentAttributes,
    OpenInferenceSpanKindValues,
    RerankerAttributes,
    SpanAttributes,
)

from .helpers import redact
from .setup import get_tracer


@contextmanager
def llm_span(
    name: str,
    model_name: str,
    messages: list[dict],
    invocation_params: dict[str, Any] | None = None,
    provider: str = "llama_cpp",
) -> Generator[Any, None, None]:
    """Reusable LLM span with automatic input capture and redaction."""
    tracer = get_tracer(__name__)

    # Redact messages for safety
    safe_messages = [
        {"role": m["role"], "content": redact(str(m.get("content", "")))}
        for m in messages
    ]

    # Extract a simple string input for the UI "Input" column
    # Usually the last user message or the whole conversation summary
    input_value_str = ""
    if messages:
        last_msg = messages[-1]
        content = last_msg.get("content", "")
        if isinstance(content, str):
            input_value_str = content
        elif isinstance(content, list):
            # Handle multimodal content lists
            input_value_str = " ".join(
                p.get("text", "") for p in content if p.get("type") == "text"
            )

    attributes = {
        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
        SpanAttributes.LLM_MODEL_NAME: model_name,
        SpanAttributes.LLM_PROVIDER: provider,
        SpanAttributes.LLM_INPUT_MESSAGES: json.dumps(
            safe_messages, ensure_ascii=False
        ),
        SpanAttributes.INPUT_VALUE: redact(input_value_str[:2000]),  # For UI column
        SpanAttributes.INPUT_MIME_TYPE: "text/plain",
    }

    if invocation_params:
        attributes[SpanAttributes.LLM_INVOCATION_PARAMETERS] = json.dumps(
            invocation_params
        )

    with tracer.start_as_current_span(name, attributes=attributes) as span:
        yield span


@contextmanager
def embedding_span(
    name: str,
    model_name: str,
    texts: list[str],
) -> Generator[Any, None, None]:
    """Reusable embedding span with indexed per-text attributes."""
    tracer = get_tracer(__name__)
    attributes = {
        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.EMBEDDING.value,
        SpanAttributes.EMBEDDING_MODEL_NAME: model_name,
        SpanAttributes.INPUT_VALUE: json.dumps(
            [redact(t) for t in texts], ensure_ascii=False
        ),
        SpanAttributes.INPUT_MIME_TYPE: "application/json",
    }
    with tracer.start_as_current_span(name, attributes=attributes) as span:
        yield span


@contextmanager
def reranker_span(
    name: str,
    model_name: str,
    query: str,
    documents: list[str],
    top_k: int,
) -> Generator[Any, None, None]:
    """Reusable reranker span with indexed input documents."""
    tracer = get_tracer(__name__)
    attributes = {
        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.RERANKER.value,
        RerankerAttributes.RERANKER_QUERY: redact(query),
        RerankerAttributes.RERANKER_MODEL_NAME: model_name,
        RerankerAttributes.RERANKER_TOP_K: top_k,
    }
    with tracer.start_as_current_span(name, attributes=attributes) as span:
        for i, doc_text in enumerate(documents):
            span.set_attribute(
                f"{RerankerAttributes.RERANKER_INPUT_DOCUMENTS}.{i}.{DocumentAttributes.DOCUMENT_CONTENT}",
                redact(str(doc_text)),
            )
        yield span


@contextmanager
def tool_span(
    name: str,
    tool_name: str,
    parameters: dict[str, Any] | None = None,
    schema_version: str | None = None,
) -> Generator[Any, None, None]:
    """Reusable tool span with parameter capture."""
    tracer = get_tracer(__name__)
    attributes = {
        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.TOOL.value,
        SpanAttributes.TOOL_NAME: tool_name,
    }
    if parameters:
        safe_params = {k: redact(str(v)) for k, v in parameters.items()}
        attributes[SpanAttributes.TOOL_PARAMETERS] = json.dumps(
            safe_params, ensure_ascii=False
        )
        # Also set INPUT_VALUE for UI visibility
        attributes[SpanAttributes.INPUT_VALUE] = json.dumps(
            safe_params, ensure_ascii=False
        )[:1000]

    if schema_version:
        attributes["tool.schema_version"] = schema_version

    with tracer.start_as_current_span(name, attributes=attributes) as span:
        yield span


@contextmanager
def agent_span(
    name: str,
    session_id: str,
    prompt_template_version: str,
    system_prompt_hash: str,
    max_steps: int,
) -> Generator[Any, None, None]:
    """Reusable agent root span with session metadata."""
    tracer = get_tracer(__name__)
    attributes = {
        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.AGENT.value,
        SpanAttributes.SESSION_ID: session_id,
        "agent.prompt_template_version": prompt_template_version,
        "agent.system_prompt_hash": system_prompt_hash,
        "agent.max_steps": max_steps,
    }
    with tracer.start_as_current_span(name, attributes=attributes) as span:
        yield span
