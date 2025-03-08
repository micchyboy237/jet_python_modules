import re
import unidecode


def fix_and_unidecode(text: str) -> str:
    """Converts escaped sequences to actual characters before unidecoding,
    while preserving direct Unicode characters."""

    # Decode only escaped sequences (e.g., "\u00e9" → "é"), keep direct Unicode intact
    def decode_match(match):
        return bytes(match.group(0), "utf-8").decode("unicode_escape")

    fixed_text = re.sub(
        r'\\u[0-9A-Fa-f]{4}|\\x[0-9A-Fa-f]{2}', decode_match, text)

    return unidecode.unidecode(fixed_text)


__all__ = [
    "fix_and_unidecode"
]
