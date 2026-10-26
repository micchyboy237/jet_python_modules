import logging
import re
from typing import Any, Dict, Tuple

from trafilatura import extract

logger = logging.getLogger(__name__)

# Matches hallucinated wrappers like https://ph.indeed.com/viewjob?jk=link_1
# Group 1 is the short link id.
_WRAPPED_PATTERN = re.compile(r"https?://[^\s\"'<>)\]]*?\b(link_\d+)\b")
_MARKDOWN_PATTERN = re.compile(r"\[([^\]]+)\]\((link_\d+)\)")
_BARE_PATTERN = re.compile(r"\blink_\d+\b")


def inflate_urls(text: str, url_map: Dict[str, str]) -> str:
    """Replaces short link IDs (e.g., link_0) with their original URLs in a string.

    Handles three forms, in this order:
    1. Wrapped:  https://site.com/path?x=link_1  -> original URL
    2. Markdown: [title](link_1)                 -> [title](original URL)
    3. Bare:     link_1                          -> original URL
    """
    if not url_map:
        return text

    def replace_wrapped(match: re.Match) -> str:
        link_id = match.group(1)
        if link_id in url_map:
            logger.debug(
                f"🔗 Inflated wrapped '{match.group(0)}' -> {url_map[link_id]}"
            )
            return url_map[link_id]
        logger.warning(f"⚠️ Unknown link id in wrapped URL: {link_id}")
        return match.group(0)

    def replace_markdown(match: re.Match) -> str:
        link_id = match.group(2)
        if link_id in url_map:
            logger.debug(f"🔗 Inflated markdown '{link_id}' -> {url_map[link_id]}")
            return f"[{match.group(1)}]({url_map[link_id]})"
        logger.warning(f"⚠️ Unknown link id in markdown link: {link_id}")
        return match.group(0)

    def replace_bare(match: re.Match) -> str:
        link_id = match.group(0)
        if link_id in url_map:
            logger.debug(f"🔗 Inflated bare '{link_id}' -> {url_map[link_id]}")
            return url_map[link_id]
        logger.warning(f"⚠️ Unknown bare link id: {link_id}")
        return link_id

    # Order matters: wrapped must run first, or the bare pattern would
    # replace link_N inside the fake URL and produce a broken one.
    text = _WRAPPED_PATTERN.sub(replace_wrapped, text)
    text = _MARKDOWN_PATTERN.sub(replace_markdown, text)
    text = _BARE_PATTERN.sub(replace_bare, text)
    return text


def inflate_urls_in_object(obj: Any, url_map: Dict[str, str]) -> Any:
    """Recursively inflate URLs in strings within a JSON object (dict/list)."""
    if not url_map:
        return obj
    if isinstance(obj, str):
        return inflate_urls(obj, url_map)
    elif isinstance(obj, list):
        return [inflate_urls_in_object(item, url_map) for item in obj]
    elif isinstance(obj, dict):
        return {k: inflate_urls_in_object(v, url_map) for k, v in obj.items()}
    else:
        return obj


def extract_content(
    html_content: str,
    url: str | None = None,
    include_links: bool = True,
    include_metadata: bool = True,
) -> Tuple[str, Dict[str, str]]:
    """
    Extract main content from HTML using Trafilatura.
    Replaces long URLs with short IDs to save tokens.

    Returns:
        A tuple of (markdown_content, url_map) where url_map maps ID -> Original URL.
    """
    result = extract(
        html_content,
        output_format="markdown",
        include_links=include_links,
        with_metadata=include_metadata,
        url=url,
    )
    if not result:
        return "", {}

    url_map: Dict[str, str] = {}
    counter = 0

    def replace_url(match):
        nonlocal counter
        full_match = match.group(0)
        text_part = match.group(1)
        url_part = match.group(2)
        if len(url_part) > 30:
            link_id = f"link_{counter}"
            counter += 1
            url_map[link_id] = url_part
            return f"[{text_part}]({link_id})"
        return full_match

    pattern = r"\[([^\]]+)\]\(([^)]+)\)"
    cleaned_result = re.sub(pattern, replace_url, result)
    return cleaned_result, url_map
