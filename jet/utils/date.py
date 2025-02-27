from datetime import datetime, date
from jet.logger import logger
from typing import Union


def to_date(value: Union[str, datetime, date]) -> date:
    """
    Converts a string, datetime, or date object to a date object.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.fromisoformat(value).date()
    except ValueError as e:
        message = f"Invalid date format: {e}"
        logger.error(message)
        raise ValueError(message)


def is_date_greater(date_str: Union[str, datetime, date], cutoff_date: Union[str, datetime, date]) -> bool:
    """
    Checks if a given date is strictly greater than the cutoff date.
    """
    return to_date(date_str) > to_date(cutoff_date)


def is_date_greater_or_equal(date_str: Union[str, datetime, date], cutoff_date: Union[str, datetime, date]) -> bool:
    """
    Checks if a given date is greater than or equal to the cutoff date.
    Ignores time components (hour, minute, second, microsecond).
    """
    return to_date(date_str) >= to_date(cutoff_date)


def is_date_lesser(date_str: Union[str, datetime, date], cutoff_date: Union[str, datetime, date]) -> bool:
    """
    Checks if a given date is lesser than the cutoff date.
    """
    return to_date(date_str) < to_date(cutoff_date)


def is_date_lesser_or_equal(date_str: Union[str, datetime, date], cutoff_date: Union[str, datetime, date]) -> bool:
    """
    Checks if a given date is lesser than or equal to the cutoff date.
    """
    return to_date(date_str) <= to_date(cutoff_date)
