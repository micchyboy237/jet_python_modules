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


def parse_date(date_str: str) -> datetime:
    """
    Parses a string into a datetime object, supporting multiple date formats including all ISO 8601 variants.

    Supported formats:
    - ISO 8601: YYYY-MM-DD, YYYY-MM-DD HH:MM:SS, YYYY-MM-DDTHH:MM:SS, YYYY-MM-DDTHH:MM:SS.ssssss, 
                YYYY-MM-DDTHH:MM:SSZ, YYYY-MM-DDTHH:MM:SS+HH:MM
    - US: MM/DD/YYYY, MM-DD-YYYY
    - European: DD/MM/YYYY, DD-MM-YYYY
    - Verbal: Month DD, YYYY (e.g., 'January 01, 2025')
    - YYYY/MM/DD, YYYYMMDD

    Args:
        date_str: String representing a date

    Returns:
        datetime: Parsed datetime object (naive, without timezone)

    Raises:
        ValueError: If the date string cannot be parsed
    """
    if not isinstance(date_str, str):
        raise ValueError("Input must be a string")

    # List of format patterns to try
    format_patterns = [
        "%Y-%m-%d",                    # 2025-07-20
        "%Y-%m-%d %H:%M:%S",          # 2025-07-20 14:30:00
        "%Y-%m-%dT%H:%M:%S",          # 2025-07-20T14:30:00
        "%Y-%m-%dT%H:%M:%S.%f",       # 2025-07-20T14:30:00.064176
        "%Y-%m-%dT%H:%M:%SZ",         # 2025-07-20T14:30:00Z
        "%Y-%m-%dT%H:%M:%S%z",        # 2025-07-20T14:30:00+00:00
        "%m/%d/%Y",                   # 07/20/2025
        "%m-%d-%Y",                   # 07-20-2025
        "%d/%m/%Y",                   # 20/07/2025
        "%d-%m-%Y",                   # 20-07-2025
        "%Y/%m/%d",                   # 2025/07/20
        "%Y%m%d",                     # 20250720
        "%B %d, %Y",                  # July 20, 2025
        "%b %d, %Y",                  # Jul 20, 2025
    ]

    # Remove any extra whitespace and normalize
    date_str = date_str.strip()

    for fmt in format_patterns:
        try:
            parsed_dt = datetime.strptime(date_str, fmt)
            # If the parsed datetime is timezone-aware, convert to naive
            if parsed_dt.tzinfo is not None:
                parsed_dt = parsed_dt.replace(tzinfo=None)
            return parsed_dt
        except ValueError:
            continue

    message = f"Cannot parse date string '{date_str}'. Supported formats include ISO 8601 (YYYY-MM-DD, YYYY-MM-DDTHH:MM:SS, YYYY-MM-DDTHH:MM:SS.ssssss, YYYY-MM-DDTHH:MM:SSZ, YYYY-MM-DDTHH:MM:SS+HH:MM), MM/DD/YYYY, DD/MM/YYYY, YYYYMMDD, or 'Month DD, YYYY'."
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
