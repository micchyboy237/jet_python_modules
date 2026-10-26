import re
from typing import Dict, Tuple

from trafilatura import extract


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

    # Regex to find markdown links: [text](url)
    # We use a function to replace each match
    def replace_url(match):
        nonlocal counter
        full_match = match.group(0)
        text_part = match.group(1)
        url_part = match.group(2)

        # Only replace if the URL is long enough to warrant it (e.g., > 30 chars)
        if len(url_part) > 30:
            link_id = f"link_{counter}"
            counter += 1
            url_map[link_id] = url_part
            return f"[{text_part}]({link_id})"
        return full_match

    # Pattern for markdown links
    pattern = r"\[([^\]]+)\]\(([^)]+)\)"
    cleaned_result = re.sub(pattern, replace_url, result)

    return cleaned_result, url_map
