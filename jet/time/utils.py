import os
from datetime import datetime, timedelta
from typing import TypedDict, Union


class FileDates(TypedDict):
    created: Union[datetime, str]
    modified: Union[datetime, str]


def get_file_dates(file_path: str, format: str = None) -> FileDates:
    """
    Reads the created and modified date and time of a file.
    :param file_path: Path to the file or folder
    :param format: Optional datetime format string. If provided, dates will be formatted as strings.
    :return: Dictionary containing created and modified datetime objects or formatted strings.
    """
    stats = os.stat(file_path)
    created_time = datetime.fromtimestamp(stats.st_ctime)
    modified_time = datetime.fromtimestamp(stats.st_mtime)

    if format:
        created_time = created_time.strftime(format)
        modified_time = modified_time.strftime(format)

    return {"created": created_time, "modified": modified_time}


def calculate_time_difference(previous_time: datetime) -> timedelta:
    """
    Calculates the time difference between now and the given datetime.
    :param previous_time: A datetime object to compare with the current time.
    :return: Time difference as a timedelta object.
    """
    now = datetime.now()
    return now - previous_time


def readable_time_difference(time_difference: timedelta) -> str:
    """
    Converts a timedelta object to a human-readable string.
    :param time_difference: Timedelta object.
    :return: Readable string representation of the time difference.
    """
    days = time_difference.days
    seconds = time_difference.seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    parts = []
    if days > 0:
        parts.append(f"{days} day{'s' if days > 1 else ''}")
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours > 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes > 1 else ''}")
    if seconds > 0:
        parts.append(f"{seconds} second{'s' if seconds > 1 else ''}")

    return ", ".join(parts) if parts else "just now"
