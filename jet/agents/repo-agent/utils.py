"""
Utility functions for logging, token estimation, and async helpers.
"""

import logging
import time
from functools import wraps


def setup_logger(name: str = "repo_agent") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logger()


def timing_decorator(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = await func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed in {time.time() - start:.2f}s")
            return result
        except Exception as e:
            logger.error(
                f"{func.__name__} failed after {time.time() - start:.2f}s: {e}"
            )
            raise

    return wrapper


def estimate_tokens(text: str) -> int:
    """Rough estimate: ~4 chars per token for code/English."""
    return len(text) // 4


def truncate_text(text: str, max_tokens: int) -> str:
    if estimate_tokens(text) <= max_tokens:
        return text
    char_limit = max_tokens * 4
    return text[:char_limit] + "\n...[truncated]"
