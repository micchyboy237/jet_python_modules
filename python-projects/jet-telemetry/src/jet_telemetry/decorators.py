"""
Summary: Streamlined tracing decorators for AI/LLM applications.
Uses arize-phoenix-otel 0.17.1+ built-in decorators with automatic
input/output capture and OpenInference semantic conventions.
Includes redaction and rich attribute setting similar to jet/observability.
Prevents large vector data from being captured in spans.
"""

import inspect
import json
from functools import wraps
from typing import Optional

from openinference.semconv.trace import (
    EmbeddingAttributes,
    RerankerAttributes,
    SpanAttributes,
)
from opentelemetry import trace as otel_trace


def _redact(text: str) -> str:
    sensitive = ["ssn", "password", "api_key", "secret", "token"]
    lower = text.lower()
    for pattern in sensitive:
        if pattern in lower:
            return "[REDACTED]"
    return text


def _get_tracer():
    """Lazily get the global tracer."""
    return otel_trace.get_tracer(__name__)


def llm(
    func=None,
    *,
    model_name: str = "unknown",
    name: Optional[str] = None,
    provider: str = "llama_cpp",
):
    """
    Decorator for LLM calls.
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
                if "invocation_params" in bound_args.arguments:
                    params = bound_args.arguments["invocation_params"]
                    if isinstance(params, dict):
                        span.set_attribute(
                            SpanAttributes.LLM_INVOCATION_PARAMETERS, json.dumps(params)
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
                # Set tool metadata if provided via decorator
                if name:
                    span.set_attribute(SpanAttributes.TOOL_NAME, name)
                if description:
                    span.set_attribute(SpanAttributes.TOOL_DESCRIPTION, description)
                # Set tool parameters from function call
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
                    # Set tool metadata if provided via decorator
                    if name:
                        span.set_attribute(SpanAttributes.TOOL_NAME, name)
                    if description:
                        span.set_attribute(SpanAttributes.TOOL_DESCRIPTION, description)
                    # Set tool parameters from function call
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
                        # Note: There's no RETRIEVER_MODEL_NAME in SpanAttributes
                        # Use custom attribute instead
                        if model_name:
                            span.set_attribute("retriever.model_name", model_name)
                        if args:
                            span.set_attribute("retrieval.query", _redact(str(args[0])))
                    result = await f(*args, **kwargs)
                    # Capture retrieval documents metadata only (no vectors/large content)
                    if span.is_recording() and result:
                        if isinstance(result, list) and len(result) > 0:
                            # Only capture count and scores, not full content
                            doc_metadata = []
                            for i, doc in enumerate(result[:10]):  # Limit to 10 docs
                                meta = {"index": i}
                                if isinstance(doc, dict):
                                    if "score" in doc:
                                        meta["score"] = doc["score"]
                                    if "id" in doc:
                                        meta["id"] = doc["id"]
                                    # Skip 'content' and 'embedding' fields to avoid large spans
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
                        # Note: There's no RETRIEVER_MODEL_NAME in SpanAttributes
                        # Use custom attribute instead
                        if model_name:
                            span.set_attribute("retriever.model_name", model_name)
                        if args:
                            span.set_attribute("retrieval.query", _redact(str(args[0])))
                    result = f(*args, **kwargs)
                    # Capture retrieval documents metadata only (no vectors/large content)
                    if span.is_recording() and result:
                        if isinstance(result, list) and len(result) > 0:
                            # Only capture count and scores, not full content
                            doc_metadata = []
                            for i, doc in enumerate(result[:10]):  # Limit to 10 docs
                                meta = {"index": i}
                                if isinstance(doc, dict):
                                    if "score" in doc:
                                        meta["score"] = doc["score"]
                                    if "id" in doc:
                                        meta["id"] = doc["id"]
                                    # Skip 'content' and 'embedding' fields to avoid large spans
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
                            # Capture text input (redacted if sensitive)
                            span.set_attribute(
                                EmbeddingAttributes.EMBEDDING_TEXT,
                                _redact(str(args[0])),
                            )
                            # Capture text length for monitoring
                            span.set_attribute(
                                "embedding.text_length", len(str(args[0]))
                            )
                    result = await f(*args, **kwargs)
                    # DO NOT capture embedding vector - it's too large
                    # Instead capture vector dimension if available
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
                            # Capture text input (redacted if sensitive)
                            span.set_attribute(
                                EmbeddingAttributes.EMBEDDING_TEXT,
                                _redact(str(args[0])),
                            )
                            # Capture text length for monitoring
                            span.set_attribute(
                                "embedding.text_length", len(str(args[0]))
                            )
                    result = f(*args, **kwargs)
                    # DO NOT capture embedding vector - it's too large
                    # Instead capture vector dimension if available
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
                        # Use RerankerAttributes.RERANKER_MODEL_NAME
                        if model_name:
                            span.set_attribute(
                                RerankerAttributes.RERANKER_MODEL_NAME, model_name
                            )
                        # Capture query if present
                        if args and len(args) > 0:
                            span.set_attribute(
                                RerankerAttributes.RERANKER_QUERY, _redact(str(args[0]))
                            )
                    result = await f(*args, **kwargs)
                    # Capture reranked documents metadata only (no full content)
                    if span.is_recording() and result:
                        if isinstance(result, list):
                            span.set_attribute(
                                "reranker.output_document_count", len(result)
                            )
                            # Only capture scores, not full content
                            scores = []
                            for doc in result[:10]:  # Limit to 10 docs
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
                        # Use RerankerAttributes.RERANKER_MODEL_NAME
                        if model_name:
                            span.set_attribute(
                                RerankerAttributes.RERANKER_MODEL_NAME, model_name
                            )
                        # Capture query if present
                        if args and len(args) > 0:
                            span.set_attribute(
                                RerankerAttributes.RERANKER_QUERY, _redact(str(args[0]))
                            )
                    result = f(*args, **kwargs)
                    # Capture reranked documents metadata only (no full content)
                    if span.is_recording() and result:
                        if isinstance(result, list):
                            span.set_attribute(
                                "reranker.output_document_count", len(result)
                            )
                            # Only capture scores, not full content
                            scores = []
                            for doc in result[:10]:  # Limit to 10 docs
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
