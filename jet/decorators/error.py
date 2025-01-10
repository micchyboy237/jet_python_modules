from typing import Callable

from jet.logger import logger
from jet.logger.timer import sleep_countdown


def wrap_retry(
    func: Callable,
    max_retries=5,
    delay=3,
):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
                sleep_countdown(delay)
            else:
                logger.error("Max retries reached. Raising the exception.")
                raise e
