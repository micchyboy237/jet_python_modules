"""
Summary: Streamlined tracing decorators for AI/LLM applications.
Uses arize-phoenix-otel 0.17.1+ built-in decorators with automatic
input/output capture and OpenInference semantic conventions.
Includes redaction and rich attribute setting similar to jet/observability.
"""

import inspect
import json
from functools import wraps
from typing import Optional

from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace as otel_trace


# Simple redaction helper to match jet/observability style
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

        # Use official decorator for basic structure
        decorated = tracer.llm(name=span_name)(f)

        @wraps(decorated)
        def wrapper(*args, **kwargs):
            # Extract potential messages/params from args/kwargs if possible
            # This is a simplified extraction; for full fidelity, use context managers like in jet/observability
            span = otel_trace.get_current_span()
            if span.is_recording():
                span.set_attribute(SpanAttributes.LLM_MODEL_NAME, model_name)
                span.set_attribute(SpanAttributes.LLM_PROVIDER, provider)

                # Try to find 'messages' in kwargs or args
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

        # Handle async if original was async
        if inspect.iscoroutinefunction(f):

            @wraps(decorated)
            async def async_wrapper(*args, **kwargs):
                span = otel_trace.get_current_span()
                # Same attribute setting logic as sync wrapper
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
            if span.is_recording() and kwargs:
                safe_params = {k: _redact(str(v)) for k, v in kwargs.items()}
                span.set_attribute(
                    SpanAttributes.TOOL_PARAMETERS, json.dumps(safe_params)
                )
            return decorated(*args, **kwargs)

        if inspect.iscoroutinefunction(f):

            @wraps(decorated)
            async def async_wrapper(*args, **kwargs):
                span = otel_trace.get_current_span()
                if span.is_recording() and kwargs:
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
            # Fallback for RETRIEVER, RERANKER, etc.
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
