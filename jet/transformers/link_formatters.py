from typing import List, Dict, Optional, TypedDict
from urllib.parse import urlparse, unquote
import re


def format_links_for_embedding(
    links: List[str],
    fragment_replacements: Optional[Dict[str, str]] = None
) -> List[str]:
    """Format links for embedding search with host (no scheme), decoded and readable paths, queries, fragments, and uniqueness.

    Args:
        links: List of URLs to format.
        fragment_replacements: Optional dictionary mapping fragment patterns to replacements for custom handling.

    Returns:
        List of formatted, unique strings with decoded and readable components, using newlines and key-value formatting.
    """

    formatted_links = set()  # Use set to ensure uniqueness
    # Unwanted patterns to filter (e.g., technical paths)
    unwanted_patterns = (
        r'wp-json|oembed|feed|xmlrpc|wp-content|wp-includes|wp-admin'
    )
    # Default fragment replacements if none provided
    fragment_replacements = fragment_replacements or {}

    for link in links:
        # Parse URL components
        parsed = urlparse(link)
        host = parsed.netloc
        path = parsed.path.strip("/")
        query = parsed.query
        fragment = parsed.fragment

        # Skip unwanted technical paths
        if path and re.search(unwanted_patterns, path, re.IGNORECASE):
            continue

        # Skip empty or root paths with no meaningful content
        if not host or (not path and not query and not fragment):
            continue

        # Format path for readability: replace hyphens, underscores, slashes with spaces, then title case
        readable_path = path.replace(
            "-", " ").replace("_", " ").replace("/", " ").title() if path else ""

        # Format fragment for readability: decode, replace hyphens/underscores/colons, and title case
        readable_fragment = fragment
        if fragment:
            readable_fragment = unquote(fragment)
            # Apply custom fragment replacements if provided
            for pattern, replacement in fragment_replacements.items():
                readable_fragment = readable_fragment.replace(
                    pattern, replacement)
            # General readability: replace hyphens, underscores, colons with spaces
            readable_fragment = readable_fragment.replace(
                "-", " ").replace("_", " ").replace(":", " ").title()

        # Format query for readability: decode and title case
        readable_query = unquote(query).title() if query else ""

        # Combine host, path, query, and fragment with newlines and keys
        formatted = f"Link: {link}\nHost: {host}"
        if readable_path:
            formatted += f"\nPath: {readable_path}"
        if readable_query:
            formatted += f"\nQuery: {readable_query}"
        if readable_fragment:
            formatted += f"\nFragment: {readable_fragment}"

        # Skip if empty after formatting
        if formatted.strip():
            formatted_links.add(formatted)

    # Return sorted list for consistent output
    return sorted(list(formatted_links))


class LinkInput(TypedDict):
    """TypedDict for link input with url and text."""
    url: str
    text: Optional[str]


class LinkFormatter:
    """A class to format links for embedding and maintain a mapping to original URLs."""

    def __init__(self, fragment_replacements: Optional[Dict[str, str]] = None):
        self.fragment_replacements = fragment_replacements or {}
        self.formatted_to_original_map: Dict[str, str] = {}

    def format_links_for_embedding(self, links: List[LinkInput | str]) -> List[str]:
        """Format links for embedding search with host (no scheme), decoded and readable paths, queries, fragments, and uniqueness.

        Args:
            links: List of URLs (str) or dictionaries with 'url' and optional 'text' keys.

        Returns:
            List of formatted strings with decoded and readable components, using newlines and key-value formatting.
            Updates the instance's formatted_to_original_map.
        """
        self.formatted_to_original_map.clear()  # Reset mapping for this run
        formatted_links = set()  # Use set to ensure uniqueness of formatted strings
        unwanted_patterns = r'wp-json|oembed|feed|xmlrpc|wp-content|wp-includes|wp-admin'

        for link_item in links:
            # Handle both string and dict inputs
            if isinstance(link_item, str):
                url = link_item
                text = None
            else:  # Assume LinkInput (dict)
                url = link_item["url"]
                text = link_item.get("text")

            parsed = urlparse(url)
            host = parsed.netloc
            path = parsed.path.strip("/")
            query = parsed.query
            fragment = parsed.fragment

            # Skip unwanted technical paths
            if path and re.search(unwanted_patterns, path, re.IGNORECASE):
                continue

            # Skip empty or root paths with no meaningful content
            if not host or (not path and not query and not fragment):
                continue

            # Format path for readability
            readable_path = path.replace(
                "-", " ").replace("_", " ").replace("/", " ").title() if path else ""

            # Format fragment for readability
            readable_fragment = fragment
            if fragment:
                readable_fragment = unquote(fragment)
                for pattern, replacement in self.fragment_replacements.items():
                    readable_fragment = readable_fragment.replace(
                        pattern, replacement)
                readable_fragment = readable_fragment.replace(
                    "-", " ").replace("_", " ").replace(":", " ").title()

            # Format query for readability
            readable_query = unquote(query).title() if query else ""

            # Combine components
            formatted = f"Link: {url}\nHost: {host}"
            if readable_path:
                formatted += f"\nPath: {readable_path}"
            if readable_query:
                formatted += f"\nQuery: {readable_query}"
            if readable_fragment:
                formatted += f"\nFragment: {readable_fragment}"
            if text:
                formatted += f"\nText: {text.title()}"

            if formatted.strip():
                formatted_links.add(formatted)
                # Store mapping of formatted string to original URL, keeping first occurrence
                if formatted not in self.formatted_to_original_map:
                    self.formatted_to_original_map[formatted] = url

        return sorted(list(formatted_links))
