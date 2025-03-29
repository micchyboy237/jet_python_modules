import re
from urllib.parse import quote


def normalize_url(url: str) -> str:
    """Removes unnecessary characters from anime URLs and trims trailing slashes."""
    return quote(url.rstrip('/'), safe=":/")


__all__ = [
    "normalize_url",
]
