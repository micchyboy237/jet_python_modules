import re


def normalize_url(url: str) -> str:
    """Removes unnecessary characters from anime URLs."""
    return re.sub(r'[^a-zA-Z0-9:/._-]', '', url.split("?")[0])  # Remove non-standard chars


__all__ = [
    "normalize_url",
]
