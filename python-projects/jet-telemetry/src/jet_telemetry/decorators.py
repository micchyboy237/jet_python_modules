"""
Summary: Streamlined tracing decorators for AI/LLM applications.
Organized by instrumentation style:
1. Core Utilities (Performance, Redaction)
2. High-Level Wrappers (arize-phoenix-otel built-ins)
3. Semantic Manual Spans (Custom OpenInference attributes)
"""

import inspect
import json
import time
from functools import wraps
from typing import Optional

from openinference.semconv.trace import (
    EmbeddingAttributes,
    RerankerAttributes,
    SpanAttributes,
)
from opentelemetry import trace as otel_trace

# ─── 1. CORE UTILITIES ───────────────────────────────────────────────────────


def _redact(text: str) -> str:
    """Redact sensitive content from text before tracing."""
    sensitive = ["ssn", "password", "api_key", "secret", "token"]
    lower = text.lower()
    for pattern in sensitive:
        if pattern in lower:
            return "[REDACTED]"
    return text


def _get_tracer():
    """Lazily get the global tracer from Phoenix TracerProvider."""
    try:
        from .setup import get_tracer_provider

        provider = get_tracer_provider()
        if provider:
            return provider.get_tracer(__name__)
    except ImportError:
        pass
    return otel_trace.get_tracer(__name__)


def performance_monitor(func=None, *, threshold_ms: float = 500):
    """
    Attaches performance metrics (duration_ms, slow) to the CURRENT active span.
    Useful for stacking with @llm, @tool, or @chain to track latency explicitly.

    Args:
        threshold_ms: If duration exceeds this, sets 'perf.slow' to True.
    """

    def decorator(f):
        is_async = inspect.iscoroutinefunction(f)

        if is_async:

            @wraps(f)
            async def async_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                try:
                    result = await f(*args, **kwargs)
                    return result
                finally:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    _attach_perf_metrics(elapsed_ms, threshold_ms, f.__name__)

            return async_wrapper
        else:

            @wraps(f)
            def sync_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                try:
                    result = f(*args, **kwargs)
                    return result
                finally:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    _attach_perf_metrics(elapsed_ms, threshold_ms, f.__name__)

            return sync_wrapper

    if func is not None:
        return decorator(func)
    return decorator


def _attach_perf_metrics(elapsed_ms: float, threshold_ms: float, name: str):
    """Helper to attach metrics to the current active span."""
    span = otel_trace.get_current_span()
    if span.is_recording():
        # Use a standard prefix for easy querying in Phoenix
        span.set_attribute(f"perf.{name}.duration_ms", round(elapsed_ms, 2))
        span.set_attribute(f"perf.{name}.slow", elapsed_ms > threshold_ms)


# ─── 2. HIGH-LEVEL WRAPPERS (arize-phoenix-otel) ─────────────────────────────


def llm(
    func=None,
    *,
    model_name: str = "unknown",
    name: Optional[str] = None,
    provider: str = "llama_cpp",
):
    """
    Decorator for LLM calls using built-in Phoenix tracer.
    Captures input messages and invocation parameters.
    """

    def decorator(f):
        tracer = _get_tracer()
        span_name = name or f.__name__
        decorated = tracer.llm(name=span_name)(f)

        @wraps(decorated)
        def wrapper(*args, **kwargs):
            span = otel_trace.get_current_span()
            if span.is_recording():
                span.set_attribute(SpanAttributes.LLM_MODEL_NAME, model_name)
                span.set_attribute(SpanAttributes.LLM_PROVIDER, provider)

                # Capture input messages if available
                sig = inspect.signature(f)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                if "messages" in bound_args.arguments:
                    msgs = bound_args.arguments["messages"]
                    if isinstance(msgs, list):
                        safe_msgs = [
                            {
                                "role": m.get("role"),
                                "content": _redact(str(m.get("content", ""))),
                            }
                            for m in msgs
                        ]
                        span.set_attribute(
                            SpanAttributes.LLM_INPUT_MESSAGES, json.dumps(safe_msgs)
                        )
            return decorated(*args, **kwargs)

        if inspect.iscoroutinefunction(f):

            @wraps(decorated)
            async def async_wrapper(*args, **kwargs):
                span = otel_trace.get_current_span()
                if span.is_recording():
                    span.set_attribute(SpanAttributes.LLM_MODEL_NAME, model_name)
                    span.set_attribute(SpanAttributes.LLM_PROVIDER, provider)

                    sig = inspect.signature(f)
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()
                    if "messages" in bound_args.arguments:
                        msgs = bound_args.arguments["messages"]
                        if isinstance(msgs, list):
                            safe_msgs = [
                                {
                                    "role": m.get("role"),
                                    "content": _redact(str(m.get("content", ""))),
                                }
                                for m in msgs
                            ]
                            span.set_attribute(
                                SpanAttributes.LLM_INPUT_MESSAGES, json.dumps(safe_msgs)
                            )
                return await decorated(*args, **kwargs)

            return async_wrapper

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def tool(func=None, *, name: Optional[str] = None, description: Optional[str] = None):
    """
    Decorator for external interactions (Vector DB, APIs, Rerankers).
    """

    def decorator(f):
        tracer = _get_tracer()
        kwargs = {}
        if name:
            kwargs["name"] = name
        if description:
            kwargs["description"] = description
        decorated = tracer.tool(**kwargs)(f)

        @wraps(decorated)
        def wrapper(*args, **kwargs):
            span = otel_trace.get_current_span()
            if span.is_recording():
                if name:
                    span.set_attribute(SpanAttributes.TOOL_NAME, name)
                if description:
                    span.set_attribute(SpanAttributes.TOOL_DESCRIPTION, description)
                if kwargs:
                    safe_params = {k: _redact(str(v)) for k, v in kwargs.items()}
                    span.set_attribute(
                        SpanAttributes.TOOL_PARAMETERS, json.dumps(safe_params)
                    )
            return decorated(*args, **kwargs)

        if inspect.iscoroutinefunction(f):

            @wraps(decorated)
            async def async_wrapper(*args, **kwargs):
                span = otel_trace.get_current_span()
                if span.is_recording():
                    if name:
                        span.set_attribute(SpanAttributes.TOOL_NAME, name)
                    if description:
                        span.set_attribute(SpanAttributes.TOOL_DESCRIPTION, description)
                    if kwargs:
                        safe_params = {k: _redact(str(v)) for k, v in kwargs.items()}
                        span.set_attribute(
                            SpanAttributes.TOOL_PARAMETERS, json.dumps(safe_params)
                        )
                return await decorated(*args, **kwargs)

            return async_wrapper

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def chain(func=None, *, name: Optional[str] = None):
    """
    Decorator for orchestration logic (RAG pipelines, Memory updates).
    """

    def decorator(f):
        tracer = _get_tracer()
        span_name = name or f.__name__
        return tracer.chain(name=span_name)(f)

    if func is not None:
        return decorator(func)
    return decorator


def agent(func=None, *, name: Optional[str] = None):
    """
    Decorator for agent reasoning loops.
    """

    def decorator(f):
        tracer = _get_tracer()
        span_name = name or f.__name__
        return tracer.agent(name=span_name)(f)

    if func is not None:
        return decorator(func)
    return decorator


# ─── 3. SEMANTIC MANUAL SPANS (Custom OpenInference) ─────────────────────────


def retriever(
    func=None, *, name: Optional[str] = None, model_name: Optional[str] = None
):
    """
    Decorator for data retrieval steps (Vector DB, SQL, etc.).
    Uses RETRIEVAL_DOCUMENTS for results but limits document content size.
    """

    def decorator(f):
        tracer = _get_tracer()
        span_name = name or f.__name__
        is_async = inspect.iscoroutinefunction(f)

        if is_async:

            @wraps(f)
            async def async_wrapper(*args, **kwargs):
                with tracer.start_as_current_span(
                    span_name, openinference_span_kind="retriever"
                ) as span:
                    if span.is_recording():
                        if model_name:
                            span.set_attribute("retriever.model_name", model_name)
                        if args:
                            span.set_attribute("retrieval.query", _redact(str(args[0])))
                    result = await f(*args, **kwargs)
                    if span.is_recording() and result:
                        if isinstance(result, list) and len(result) > 0:
                            doc_metadata = []
                            for i, doc in enumerate(result[:10]):
                                meta = {"index": i}
                                if isinstance(doc, dict):
                                    if "score" in doc:
                                        meta["score"] = doc["score"]
                                    if "id" in doc:
                                        meta["id"] = doc["id"]
                                doc_metadata.append(meta)
                            span.set_attribute("retrieval.document_count", len(result))
                            span.set_attribute(
                                "retrieval.documents.metadata", json.dumps(doc_metadata)
                            )
                    return result

            return async_wrapper
        else:

            @wraps(f)
            def sync_wrapper(*args, **kwargs):
                with tracer.start_as_current_span(
                    span_name, openinference_span_kind="retriever"
                ) as span:
                    if span.is_recording():
                        if model_name:
                            span.set_attribute("retriever.model_name", model_name)
                        if args:
                            span.set_attribute("retrieval.query", _redact(str(args[0])))
                    result = f(*args, **kwargs)
                    if span.is_recording() and result:
                        if isinstance(result, list) and len(result) > 0:
                            doc_metadata = []
                            for i, doc in enumerate(result[:10]):
                                meta = {"index": i}
                                if isinstance(doc, dict):
                                    if "score" in doc:
                                        meta["score"] = doc["score"]
                                    if "id" in doc:
                                        meta["id"] = doc["id"]
                                doc_metadata.append(meta)
                            span.set_attribute("retrieval.document_count", len(result))
                            span.set_attribute(
                                "retrieval.documents.metadata", json.dumps(doc_metadata)
                            )
                    return result

            return sync_wrapper

    if func is not None:
        return decorator(func)
    return decorator


def embedding(func=None, *, name: Optional[str] = None, model_name: str = "unknown"):
    """
    Decorator for embedding generation.
    Captures text input but NOT the embedding vector to avoid large spans.
    """

    def decorator(f):
        tracer = _get_tracer()
        span_name = name or f.__name__
        is_async = inspect.iscoroutinefunction(f)

        if is_async:

            @wraps(f)
            async def async_wrapper(*args, **kwargs):
                with tracer.start_as_current_span(
                    span_name, openinference_span_kind="embedding"
                ) as span:
                    if span.is_recording():
                        span.set_attribute(
                            SpanAttributes.EMBEDDING_MODEL_NAME, model_name
                        )
                        if args:
                            span.set_attribute(
                                EmbeddingAttributes.EMBEDDING_TEXT,
                                _redact(str(args[0])),
                            )
                            span.set_attribute(
                                "embedding.text_length", len(str(args[0]))
                            )
                    result = await f(*args, **kwargs)
                    if span.is_recording() and result:
                        if isinstance(result, list):
                            span.set_attribute(
                                "embedding.vector_dimension", len(result)
                            )
                    return result

            return async_wrapper
        else:

            @wraps(f)
            def sync_wrapper(*args, **kwargs):
                with tracer.start_as_current_span(
                    span_name, openinference_span_kind="embedding"
                ) as span:
                    if span.is_recording():
                        span.set_attribute(
                            SpanAttributes.EMBEDDING_MODEL_NAME, model_name
                        )
                        if args:
                            span.set_attribute(
                                EmbeddingAttributes.EMBEDDING_TEXT,
                                _redact(str(args[0])),
                            )
                            span.set_attribute(
                                "embedding.text_length", len(str(args[0]))
                            )
                    result = f(*args, **kwargs)
                    if span.is_recording() and result:
                        if isinstance(result, list):
                            span.set_attribute(
                                "embedding.vector_dimension", len(result)
                            )
                    return result

            return sync_wrapper

    if func is not None:
        return decorator(func)
    return decorator


def reranker(
    func=None, *, name: Optional[str] = None, model_name: Optional[str] = None
):
    """
    Decorator for document reranking.
    Uses RerankerAttributes for proper semantic conventions.
    Captures metadata only, not full document content.
    """

    def decorator(f):
        tracer = _get_tracer()
        span_name = name or f.__name__
        is_async = inspect.iscoroutinefunction(f)

        if is_async:

            @wraps(f)
            async def async_wrapper(*args, **kwargs):
                with tracer.start_as_current_span(
                    span_name, openinference_span_kind="reranker"
                ) as span:
                    if span.is_recording():
                        if model_name:
                            span.set_attribute(
                                RerankerAttributes.RERANKER_MODEL_NAME, model_name
                            )
                        if args and len(args) > 0:
                            span.set_attribute(
                                RerankerAttributes.RERANKER_QUERY, _redact(str(args[0]))
                            )
                    result = await f(*args, **kwargs)
                    if span.is_recording() and result:
                        if isinstance(result, list):
                            span.set_attribute(
                                "reranker.output_document_count", len(result)
                            )
                            scores = []
                            for doc in result[:10]:
                                if isinstance(doc, dict) and "score" in doc:
                                    scores.append(doc["score"])
                            if scores:
                                span.set_attribute(
                                    "reranker.output_scores", json.dumps(scores)
                                )
                    return result

            return async_wrapper
        else:

            @wraps(f)
            def sync_wrapper(*args, **kwargs):
                with tracer.start_as_current_span(
                    span_name, openinference_span_kind="reranker"
                ) as span:
                    if span.is_recording():
                        if model_name:
                            span.set_attribute(
                                RerankerAttributes.RERANKER_MODEL_NAME, model_name
                            )
                        if args and len(args) > 0:
                            span.set_attribute(
                                RerankerAttributes.RERANKER_QUERY, _redact(str(args[0]))
                            )
                    result = f(*args, **kwargs)
                    if span.is_recording() and result:
                        if isinstance(result, list):
                            span.set_attribute(
                                "reranker.output_document_count", len(result)
                            )
                            scores = []
                            for doc in result[:10]:
                                if isinstance(doc, dict) and "score" in doc:
                                    scores.append(doc["score"])
                            if scores:
                                span.set_attribute(
                                    "reranker.output_scores", json.dumps(scores)
                                )
                    return result

            return sync_wrapper

    if func is not None:
        return decorator(func)
    return decorator


def guardrail(func=None, *, name: Optional[str] = None):
    """
    Decorator for safety and policy validation.
    """

    def decorator(f):
        tracer = _get_tracer()
        span_name = name or f.__name__
        is_async = inspect.iscoroutinefunction(f)

        if is_async:

            @wraps(f)
            async def async_wrapper(*args, **kwargs):
                with tracer.start_as_current_span(
                    span_name, openinference_span_kind="guardrail"
                ) as span:
                    if span.is_recording() and args:
                        span.set_attribute("guardrail.input", _redact(str(args[0])))
                    result = await f(*args, **kwargs)
                    if span.is_recording():
                        span.set_attribute("guardrail.result", str(result))
                    return result

            return async_wrapper
        else:

            @wraps(f)
            def sync_wrapper(*args, **kwargs):
                with tracer.start_as_current_span(
                    span_name, openinference_span_kind="guardrail"
                ) as span:
                    if span.is_recording() and args:
                        span.set_attribute("guardrail.input", _redact(str(args[0])))
                    result = f(*args, **kwargs)
                    if span.is_recording():
                        span.set_attribute("guardrail.result", str(result))
                    return result

            return sync_wrapper

    if func is not None:
        return decorator(func)
    return decorator


def evaluator(func=None, *, name: Optional[str] = None):
    """
    Decorator for evaluation logic (scoring, feedback).
    """

    def decorator(f):
        tracer = _get_tracer()
        span_name = name or f.__name__
        is_async = inspect.iscoroutinefunction(f)

        if is_async:

            @wraps(f)
            async def async_wrapper(*args, **kwargs):
                with tracer.start_as_current_span(
                    span_name, openinference_span_kind="evaluator"
                ) as span:
                    return await f(*args, **kwargs)

            return async_wrapper
        else:

            @wraps(f)
            def sync_wrapper(*args, **kwargs):
                with tracer.start_as_current_span(
                    span_name, openinference_span_kind="evaluator"
                ) as span:
                    return f(*args, **kwargs)

            return sync_wrapper

    if func is not None:
        return decorator(func)
    return decorator


def prompt(func=None, *, name: Optional[str] = None):
    """
    Decorator for prompt templating/rendering.
    """

    def decorator(f):
        tracer = _get_tracer()
        span_name = name or f.__name__
        is_async = inspect.iscoroutinefunction(f)

        if is_async:

            @wraps(f)
            async def async_wrapper(*args, **kwargs):
                with tracer.start_as_current_span(
                    span_name, openinference_span_kind="prompt"
                ) as span:
                    return await f(*args, **kwargs)

            return async_wrapper
        else:

            @wraps(f)
            def sync_wrapper(*args, **kwargs):
                with tracer.start_as_current_span(
                    span_name, openinference_span_kind="prompt"
                ) as span:
                    return f(*args, **kwargs)

            return sync_wrapper

    if func is not None:
        return decorator(func)
    return decorator


def trace(func=None, *, name: Optional[str] = None, kind: str = "CHAIN"):
    """
    Generic decorator for any function with customizable span kind.
    """

    def decorator(f):
        tracer = _get_tracer()
        span_name = name or f.__name__
        kind_upper = kind.upper()

        if kind_upper == "CHAIN":
            return tracer.chain(name=span_name)(f)
        elif kind_upper == "TOOL":
            return tracer.tool(name=span_name)(f)
        elif kind_upper == "LLM":
            return tracer.llm(name=span_name)(f)
        elif kind_upper == "AGENT":
            return tracer.agent(name=span_name)(f)
        else:
            is_async = inspect.iscoroutinefunction(f)
            if is_async:

                @wraps(f)
                async def async_wrapper(*args, **kwargs):
                    with tracer.start_as_current_span(
                        span_name,
                        openinference_span_kind=kind.lower(),
                    ) as span:
                        return await f(*args, **kwargs)

                return async_wrapper
            else:

                @wraps(f)
                def sync_wrapper(*args, **kwargs):
                    with tracer.start_as_current_span(
                        span_name,
                        openinference_span_kind=kind.lower(),
                    ) as span:
                        return f(*args, **kwargs)

                return sync_wrapper

    if func is not None:
        return decorator(func)
    return decorator
