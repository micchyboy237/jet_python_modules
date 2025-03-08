def format_time(duration: float) -> str:
    """
    Converts a duration in seconds to a formatted string.
    - If the duration is less than 1 minute, it's formatted in seconds (e.g., 59s).
    - If the duration is between 1 minute and 1 hour, it's formatted as minutes and seconds (e.g., 1m 30s).
    - If the duration is greater than 1 hour, it's formatted as hours, minutes, and seconds (e.g., 1h 30m 45s).
    """
    hours, remainder = divmod(int(duration), 3600)
    minutes, seconds = divmod(remainder, 60)

    time_parts = []

    if hours:
        time_parts.append(f"{hours}h")
    if minutes:
        time_parts.append(f"{minutes}m")
    if seconds or not time_parts:  # Always show seconds if there's no other part
        time_parts.append(f"{seconds}s")

    return " ".join(time_parts)
