"""
Summary: Streamlined tracing decorators for AI/LLM applications.
Updated for arize-phoenix-otel 0.17.1+ and OTEL 1.44.0+.
Uses standard OpenTelemetry Tracer and OpenInference semantic conventions.
"""

from functools import wraps
from typing import Any, Callable, Optional

from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace as otel_trace

_tracer = otel_trace.get_tracer(__name__)


def _set_attribute(span, key: str, value: Any):
    """Helper to safely set attributes on a span."""
    if span and span.is_recording() and value is not None:
        span.set_attribute(key, value)


def llm(func: Optional[Callable] = None, *, model_name: str = "unknown"):
    """Decorator for LLM calls."""

    def decorator(f):
        @wraps(f)
        async def wrapper(*args, **kwargs):
            with _tracer.start_as_current_span(f.__name__) as span:
                _set_attribute(
                    span,
                    SpanAttributes.OPENINFERENCE_SPAN_KIND,
                    OpenInferenceSpanKindValues.LLM.value,
                )
                _set_attribute(span, SpanAttributes.LLM_MODEL_NAME, model_name)
                return await f(*args, **kwargs)

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def tool(func: Optional[Callable] = None):
    """Decorator for external interactions (Vector DB, APIs, Rerankers)."""

    def decorator(f):
        @wraps(f)
        async def wrapper(*args, **kwargs):
            with _tracer.start_as_current_span(f.__name__) as span:
                _set_attribute(
                    span,
                    SpanAttributes.OPENINFERENCE_SPAN_KIND,
                    OpenInferenceSpanKindValues.TOOL.value,
                )
                return await f(*args, **kwargs)

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def chain(func: Optional[Callable] = None):
    """Decorator for orchestration logic (RAG pipelines, Memory updates)."""

    def decorator(f):
        @wraps(f)
        async def wrapper(*args, **kwargs):
            with _tracer.start_as_current_span(f.__name__) as span:
                _set_attribute(
                    span,
                    SpanAttributes.OPENINFERENCE_SPAN_KIND,
                    OpenInferenceSpanKindValues.CHAIN.value,
                )
                return await f(*args, **kwargs)

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def trace(func: Optional[Callable] = None, *, name: Optional[str] = None):
    """Generic decorator for any function."""

    def decorator(f):
        span_name = name or f.__name__

        @wraps(f)
        async def wrapper(*args, **kwargs):
            with _tracer.start_as_current_span(span_name):
                return await f(*args, **kwargs)

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator
