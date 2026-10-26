import re
from typing import Any, Dict, Tuple

from trafilatura import extract


def inflate_urls(text: str, url_map: Dict[str, str]) -> str:
    """Replaces short link IDs (e.g., link_0) with their original URLs in a string."""
    if not url_map:
        return text

    def replace_id(match):
        link_id = match.group(2)
        if link_id in url_map:
            return f"[{match.group(1)}]({url_map[link_id]})"
        return match.group(0)

    pattern = r"\[([^\]]+)\]\((link_\d+)\)"
    return re.sub(pattern, replace_id, text)


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
