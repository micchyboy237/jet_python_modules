import traceback

from functools import wraps
from typing import Callable

from jet.logger import logger
from jet.logger.timer import sleep_countdown
from jet.transformers.formatters import format_json
from jet.transformers.object import make_serializable
from jet.utils.class_utils import get_class_name


class LoggedException(Exception):
    def __init__(self, original_exception, *args, **kwargs):
        super().__init__(str(original_exception))
        self.original_exception = original_exception
        self.args_data = args
        self.kwargs_data = kwargs


def log_exceptions(*exception_types, raise_exception=True):
    """Decorator to log unhandled exceptions and either re-raise or return an Exception instance.

    Args:
        *exception_types: Specific exception types to catch. If empty, catches all exceptions.
        raise_exception (bool): If True, re-raises the exception. If False, returns an instance of LoggedException.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)

                # Handle generators separately
                if hasattr(result, "__iter__") and not isinstance(result, (str, bytes, dict, list, tuple)):
                    return _handle_generator(result, args, kwargs, exception_types, raise_exception)

                return result
            except exception_types or Exception as exc:
                _log_exception(exc)

                if raise_exception:
                    raise

                return LoggedException(exc, *args, **kwargs)

        return wrapper

    def _handle_generator(gen, args, kwargs, exception_types, raise_exception):
        """Handles exceptions inside generators."""
        try:
            for item in gen:
                yield item
        except exception_types or Exception as exc:
            _log_exception(exc)
            if raise_exception:
                raise
            yield LoggedException(exc, *args, **kwargs)

    def _log_exception(exc):
        """Logs exception details."""
        logger.error(format_json(make_serializable(exc)))
        logger.gray(traceback.format_exc())
        logger.warning(f"Global: Handled {get_class_name(exc)}")

    return decorator


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
