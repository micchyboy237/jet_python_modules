import traceback

from functools import wraps
from typing import Callable, Generator, Any, Type

from jet.logger import logger
from jet.logger.timer import sleep_countdown
from jet.transformers.formatters import format_json
from jet.transformers.object import make_serializable
from jet.utils.class_utils import get_class_name


class LoggedException(Exception):
    """A wrapper for logging exceptions while preserving original exception data."""

    def __init__(self, original_exception: Exception, *args, **kwargs):
        super().__init__(str(original_exception))
        self.original_exception = original_exception
        self.args_data = args
        self.kwargs_data = kwargs


def log_exceptions(*exception_types: Type[Exception], raise_exception: bool = True):
    """Decorator to log unhandled exceptions and either re-raise or return a LoggedException.

    Args:
        *exception_types (Type[Exception]): Exception types to catch. If empty, catches all exceptions.
        raise_exception (bool): If True, re-raises the exception. If False, returns a LoggedException instance.
    """
    # If no exception types are passed, default to catching all exceptions
    if not exception_types:
        # Catch all if no specific types provided
        exception_types = (Exception,)

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)

                # If function returns a generator, handle exceptions inside it
                if isinstance(result, Generator):
                    return _handle_generator(result, args, kwargs, exception_types, raise_exception)

                return result
            except exception_types as exc:
                _log_exception(exc)

                if raise_exception:
                    raise

                return LoggedException(exc, *args, **kwargs)

        return wrapper

    def _handle_generator(gen: Generator, args, kwargs, exception_types, raise_exception):
        """Handles exceptions inside a generator."""
        try:
            for item in gen:
                yield item
        except exception_types as exc:
            _log_exception(exc)
            if raise_exception:
                raise
            yield LoggedException(exc, *args, **kwargs)

    def _log_exception(exc: Exception):
        """Logs exception details in different formats."""
        logger.error(format_json(make_serializable(exc)))
        logger.gray(traceback.format_exc())
        logger.warning(f"Global: Handled {get_class_name(exc)}")

    # Return decorator even if exception_types is empty
    return decorator if exception_types else decorator


def wrap_retry(
    func: Callable,
    max_retries=3,
    delay=3,
):
    """Retries a function on failure, logging exceptions and applying exponential backoff."""

    # Wrap the function with log_exceptions
    wrapped_func = log_exceptions()(func)

    for attempt in range(max_retries):
        try:
            return wrapped_func()  # This would fail if the function requires arguments
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds..."
                )
                sleep_countdown(delay)
            else:
                logger.error(
                    f"Max retries ({max_retries}) reached. Raising the exception."
                )
                raise  # Reraise after max retries
