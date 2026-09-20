"""
Summary: Streamlined tracing decorators for AI/LLM applications.
Reduced to 4 core decorators to minimize complexity while covering
all OpenInference semantic conventions (LLM, TOOL, CHAIN, GENERIC).
"""

from functools import wraps
from typing import Any, Callable, Optional

from openinference.semconv.trace import SpanAttributes
from phoenix.trace import get_current_span, traceable


def _set_attribute(key: str, value: Any):
    """Helper to safely set attributes on the current span."""
    span = get_current_span()
    if span and value is not None:
        span.set_attribute(key, value)


def llm(func: Optional[Callable] = None, *, model_name: str = "unknown"):
    """
    Decorator for LLM calls.
    Usage: @llm(model_name="gpt-4o")
    """

    def decorator(f):
        @wraps(f)
        @traceable(span_type="LLM")
        async def wrapper(*args, **kwargs):
            _set_attribute(SpanAttributes.LLM_MODEL_NAME, model_name)
            return await f(*args, **kwargs)

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def tool(func: Optional[Callable] = None):
    """
    Decorator for external interactions (Vector DB, APIs, Rerankers, Calculators).
    Usage: @tool
    """

    def decorator(f):
        @wraps(f)
        @traceable(span_type="TOOL")
        async def wrapper(*args, **kwargs):
            return await f(*args, **kwargs)

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def chain(func: Optional[Callable] = None):
    """
    Decorator for orchestration logic (RAG pipelines, Memory updates, Agents).
    Usage: @chain
    """

    def decorator(f):
        @wraps(f)
        @traceable(span_type="CHAIN")
        async def wrapper(*args, **kwargs):
            return await f(*args, **kwargs)

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def trace(func: Optional[Callable] = None, *, name: Optional[str] = None):
    """
    Generic decorator for any function.
    Usage: @trace or @trace(name="custom_step")
    """

    def decorator(f):
        span_name = name or f.__name__

        @wraps(f)
        @traceable(name=span_name)
        async def wrapper(*args, **kwargs):
            return await f(*args, **kwargs)

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator
