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
    """Reusable LLM span with automatic input capture and redaction [[13]]."""
    tracer = get_tracer(__name__)
    safe_messages = [
        {"role": m["role"], "content": redact(m["content"])} for m in messages
    ]
    attributes = {
        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
        SpanAttributes.LLM_MODEL_NAME: model_name,
        SpanAttributes.LLM_PROVIDER: provider,
        SpanAttributes.LLM_INPUT_MESSAGES: json.dumps(safe_messages),
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
    """Reusable embedding span with indexed per-text attributes [[13]]."""
    tracer = get_tracer(__name__)
    attributes = {
        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.EMBEDDING.value,
        SpanAttributes.EMBEDDING_MODEL_NAME: model_name,
        SpanAttributes.INPUT_VALUE: json.dumps([redact(t) for t in texts]),
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
    """Reusable reranker span with indexed input documents [[13]]."""
    tracer = get_tracer(__name__)
    attributes = {
        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.RERANKER.value,
        RerankerAttributes.RERANKER_QUERY: redact(query),
        RerankerAttributes.RERANKER_MODEL_NAME: model_name,
        RerankerAttributes.RERANKER_TOP_K: top_k,
    }
    with tracer.start_as_current_span(name, attributes=attributes) as span:
        # Set indexed input documents per OpenInference spec
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
    """Reusable tool span with parameter capture [[13]]."""
    tracer = get_tracer(__name__)
    attributes = {
        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.TOOL.value,
        SpanAttributes.TOOL_NAME: tool_name,
    }
    if parameters:
        safe_params = {k: redact(str(v)) for k, v in parameters.items()}
        attributes[SpanAttributes.TOOL_PARAMETERS] = json.dumps(safe_params)
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
    """Reusable agent root span with session metadata [[13]]."""
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
