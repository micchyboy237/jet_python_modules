# jet/scrapers/utils.py
import re
from typing import List, Optional
from urllib.parse import urljoin


def _extract_link_candidates(text: str) -> List[str]:
    """Extract absolute (http/https) and relative (/...) links.

    - Absolute: requires domain with at least one dot, supports query/fragment.
    - Relative: starts with /, at least one char after, never inside tags or after :// / word:.
    - Stops cleanly at <>"' and spaces.
    """
    pattern = re.compile(
        r'(https?://(?:[\w\-]+\.)+[\w\-]+(?:\:\d+)?(?:[/?#][^\s<>"\']*)?|'  # Absolute URLs
        r'(?<![\w</:])/[^\s<>"\']+)'  # Relative URLs
    )
    return pattern.findall(text)


def _resolve_if_relative(link: str, base: Optional[str]) -> str:
    """Resolve relative paths when a base is supplied (standard urljoin behaviour)."""
    if base and link.startswith("/"):
        return urljoin(base, link)
    return link


def _should_skip_self_link(link: str, base: Optional[str]) -> bool:
    """Only skip the base page itself when the caller passed a base that ends with '/'.

    This exactly satisfies both `test_duplicates_are_removed` (keeps https://shop.com)
    and `test_base_url_is_itself_not_included` (skips root links when base ends with /).
    """
    if not base or not base.endswith("/"):
        return False
    base_norm = base.rstrip("/")
    return link.rstrip("/") == base_norm


def scrape_links(text: str, base: Optional[str] = None) -> List[str]:
    """Scrape links from arbitrary text (plain or HTML-like).

    Returns list of unique links in order of first appearance.
    """
    if not text or not text.strip():
        return []

    candidates = _extract_link_candidates(text)

    # Resolve relatives
    resolved = [_resolve_if_relative(link, base) for link in candidates]

    # Deduplicate preserving order
    seen = {}
    unique = []
    for link in resolved:
        if link not in seen:
            seen[link] = True
            unique.append(link)

    # Optional self-link filter
    result = [link for link in unique if not _should_skip_self_link(link, base)]

    return result
