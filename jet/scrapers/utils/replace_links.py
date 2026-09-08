from __future__ import annotations

import logging
import re
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Matches root-relative paths starting with /
# Positive lookbehind ensures / is preceded by legitimate context chars:
#   start-of-string, whitespace, quotes, =, (, [, >
# This avoids matching inside absolute URLs (://) or HTML closing tags (</tag>)
# Stop chars prevent capturing beyond link boundaries
_ROOT_RELATIVE_PATTERN = re.compile(r'(?:(?<=^)|(?<=[\s"\'=\(\[>]))(/[^\s<>"\'\[\]]*)')


def _parse_base_host(base: str) -> str | None:
    """Extract scheme + netloc from base URL.

    Args:
        base: Full URL string.

    Returns:
        'https://example.com' style prefix, or None if invalid.
    """
    try:
        parsed = urlparse(base)
        if parsed.scheme and parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}"
        logger.warning("Base URL missing scheme or host: %s", base)
        return None
    except Exception as exc:
        logger.error("Failed to parse base URL '%s': %s", base, exc)
        return None


def replace_links(text: str, base: str) -> str:
    """Find root-relative links (/...) and prepend the parsed host from base.

    Only links starting with '/' in valid contexts are modified.
    Absolute URLs and HTML structure remain untouched.

    Args:
        text: Raw text containing potential root-relative links.
        base: URL whose scheme+host will be prepended (e.g., 'https://example.com/docs').
              Path component is ignored since /links resolve against domain root.

    Returns:
        Text with root-relative links replaced by absolute URLs.
        Returns original text unchanged if base is invalid or text is empty.
    """
    if not text or not text.strip():
        logger.debug("replace_links: empty text provided")
        return text or ""

    host_prefix = _parse_base_host(base)
    if not host_prefix:
        logger.warning("replace_links: returning original text due to invalid base")
        return text

    count = 0

    def _replacer(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return f"{host_prefix}{match.group(1)}"

    result = _ROOT_RELATIVE_PATTERN.sub(_replacer, text)
    logger.info(
        "replace_links: replaced %d root-relative link(s) using base '%s'",
        count,
        host_prefix,
    )
    return result
