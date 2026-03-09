from __future__ import annotations

import re
from fnmatch import translate
from typing import Iterable, List, Pattern


def _compile_patterns(patterns: Iterable[str]) -> List[Pattern[str]]:
    """
    Compile glob-style URL patterns into regex patterns.

    Args:
        patterns: Iterable of pattern strings.

    Returns:
        List of compiled regex patterns.
    """
    compiled: List[Pattern[str]] = []

    for pattern in patterns:
        regex = translate(pattern)
        compiled.append(re.compile(regex))

    return compiled


def _matches_any(url: str, compiled_patterns: Iterable[Pattern[str]]) -> bool:
    """
    Check whether a URL matches any compiled pattern.

    Args:
        url: URL string to test.
        compiled_patterns: Compiled regex patterns.

    Returns:
        True if a match exists.
    """
    for pattern in compiled_patterns:
        if pattern.match(url):
            return True

    return False


def filter_links(urls: Iterable[str], patterns: Iterable[str]) -> List[str]:
    """
    Filter URLs by matching against provided patterns.

    Patterns use glob syntax (fnmatch-style):
        *  matches any characters
        ?  matches single character

    Example:
        patterns = ["https://*.example.com/*", "*login*"]

    Args:
        urls: Iterable of URLs.
        patterns: Iterable of glob pattern strings.

    Returns:
        List of URLs matching at least one pattern.
    """
    compiled_patterns = _compile_patterns(patterns)

    result: List[str] = []

    for url in urls:
        if _matches_any(url, compiled_patterns):
            result.append(url)

    return result
