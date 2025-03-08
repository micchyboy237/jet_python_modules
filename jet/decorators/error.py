import traceback
import logging

from fastapi.responses import JSONResponse
from functools import wraps
from typing import Callable

from jet.logger import logger
from jet.logger.timer import sleep_countdown


def log_exceptions(func):
    """Decorator to log unhandled exceptions and return a JSON error response."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            logger.error(f"Unhandled exception: {traceback.format_exc()}")
            logger.warning(f"Global: Handled {exc.__class__.__name__}")
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal Server Error"}
            )
    return wrapper


def wrap_retry(
    func: Callable,
    max_retries=3,
    delay=3,
):
    """Retries a function on failure, logging exceptions and applying exponential backoff."""

    # Wrap the function with log_exceptions
    wrapped_func = log_exceptions(func)

    for attempt in range(max_retries):
        try:
            return wrapped_func()  # Execute the wrapped function
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds..."
                )
                sleep_countdown(delay)
            else:
                logger.error(
                    f"Max retries ({max_retries}) reached. Raising the exception.")
                raise  # Reraise after max retries


__all__ = [
    "log_exceptions",
    "wrap_retry",
]
