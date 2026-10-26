import re
from typing import Any, Dict, Tuple

from trafilatura import extract


def inflate_urls(text: str, url_map: Dict[str, str]) -> str:
    """Replaces short link IDs (e.g., link_0) with their original URLs in a string."""
    if not url_map:
        return text

    # Pattern 1: Markdown links [Text](link_X)
    def replace_markdown(match):
        link_id = match.group(2)
        if link_id in url_map:
            return f"[{match.group(1)}]({url_map[link_id]})"
        return match.group(0)

    # Pattern 2: Bare link IDs (link_X) - useful for JSON values
    def replace_bare(match):
        link_id = match.group(0)
        if link_id in url_map:
            return url_map[link_id]
        return link_id

    # Apply Markdown replacement first
    markdown_pattern = r"\[([^\]]+)\]\((link_\d+)\)"
    cleaned_text = re.sub(markdown_pattern, replace_markdown, text)

    # Apply bare ID replacement
    # We use word boundaries \b to avoid matching partial words like 'my_link_1'
    bare_pattern = r"\blink_\d+\b"
    final_text = re.sub(bare_pattern, replace_bare, cleaned_text)

    return final_text


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
