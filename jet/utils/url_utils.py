from unidecode import unidecode
from typing import List
from urllib.parse import urlparse
from typing import List, Optional
from urllib.parse import unquote, urlparse
import requests
import xml.etree.ElementTree as ET
import re


def get_sitemap_url_from_robots(base_url: str) -> str | None:
    try:
        robots_url = f"{base_url.rstrip('/')}/robots.txt"
        res = requests.get(robots_url, timeout=5)
        res.raise_for_status()
        match = re.search(r"Sitemap:\s*(.+)", res.text, re.IGNORECASE)
        return match.group(1).strip() if match else None
    except requests.RequestException as e:
        print(f"[ERROR] Failed to fetch robots.txt: {e}")
        return None


def parse_sitemap_recursive(url: str, collected: set = None) -> list:
    collected = collected or set()

    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        root = ET.fromstring(res.content)

        # XML namespaces
        ns = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

        for sitemap in root.findall("ns:sitemap", ns):
            loc = sitemap.find("ns:loc", ns)
            if loc is not None:
                parse_sitemap_recursive(loc.text.strip(), collected)

        for url_entry in root.findall("ns:url", ns):
            loc = url_entry.find("ns:loc", ns)
            if loc is not None:
                collected.add(loc.text.strip())

    except Exception as e:
        print(f"[ERROR] Failed to parse sitemap {url}: {e}")

    return sorted(collected)


def get_all_sitemap_urls(base_url: str) -> list:
    root_sitemap_url = get_sitemap_url_from_robots(base_url)
    if not root_sitemap_url:
        print("[WARN] No sitemap found in robots.txt")
        return []

    print(f"[INFO] Root sitemap: {root_sitemap_url}")
    return parse_sitemap_recursive(root_sitemap_url)


def normalize_url(url: str, base_url: Optional[str] = None) -> str:
    if not url:
        return ""

    non_crawlable = ".php" in url
    if non_crawlable:
        return url

    parsed = urlparse(url)

    # If the URL is relative, append it to base URL
    if base_url:
        if not parsed.scheme or not parsed.netloc:
            return f"{base_url.rstrip('/')}/{url.lstrip('/')}"

    normalized_url = parsed.scheme + "://" + parsed.netloc + parsed.path
    return unquote(normalized_url.rstrip('/'))


def clean_url(url: str) -> str:
    """Clean a URL by ensuring a scheme, removing trailing slashes, normalizing format, and transliterating non-ASCII to ASCII."""
    if not url:
        return ""

    # Remove leading/trailing whitespace
    url = url.strip()

    # Ensure scheme (default to https)
    if not re.match(r'^[a-zA-Z]+://', url):
        url = f'https://{url}'

    # Parse URL to normalize components
    try:
        parsed = urlparse(url)
        # Normalize path by replacing multiple slashes with a single slash and decoding
        path = re.sub(r'/+', '/', unquote(parsed.path.rstrip('/')))
        # Transliterate non-ASCII characters in path to ASCII
        path = unidecode(path) if path else path

        # Process query parameters individually
        query = ''
        if parsed.query:
            query_parts = parsed.query.split('&')
            processed_parts = []
            for part in query_parts:
                if part:
                    if '=' in part:
                        key, value = part.split('=', 1)
                        # Decode and transliterate key and value
                        processed_key = unidecode(unquote(key))
                        processed_value = unidecode(
                            unquote(value)) if value else ''
                        processed_parts.append(
                            f'{processed_key}={processed_value}' if value else processed_key)
                    else:
                        # Handle query parameters without values
                        processed_parts.append(unidecode(unquote(part)))
            query = f'?{"&".join(processed_parts)}' if processed_parts else ''

        # Decode and transliterate fragment
        fragment = f'#{unidecode(unquote(parsed.fragment))}' if parsed.fragment else ''

        # Reconstruct URL with lowercase scheme
        cleaned_url = f'{parsed.scheme.lower()}://{parsed.netloc}{path}{query}{fragment}'
        return cleaned_url
    except ValueError as e:
        raise ValueError(f"Invalid URL: {url}") from e


def parse_url(url: str) -> List[str]:
    """Convert a URL into a list of tokens for n-gram processing."""
    # Use clean_url to normalize input (decodes and transliterates)
    cleaned_url = clean_url(url)
    if not cleaned_url:
        return []

    parsed = urlparse(cleaned_url)
    tokens = []

    # Add scheme (e.g., http, https)
    if parsed.scheme:
        tokens.append(parsed.scheme)

    # Add domain components, excluding port
    if parsed.netloc:
        # Remove port if present (e.g., example.com:8080 -> example.com)
        netloc = parsed.netloc.split(
            ':')[0] if ':' in parsed.netloc else parsed.netloc
        domain_parts = netloc.split('.')
        tokens.extend([part for part in domain_parts if part])

    # Add path segments
    if parsed.path and parsed.path != '/':
        path_segments = parsed.path.strip('/').split('/')
        # Replace hyphens and underscores with spaces in path segments
        tokens.extend([segment.replace('-', ' ').replace('_', ' ')
                      for segment in path_segments if segment])

    # Add query parameters, preserving duplicates
    if parsed.query:
        # Split query string manually to preserve all values
        query_parts = parsed.query.split('&')
        for part in query_parts:
            if part:  # Skip empty parts
                if '=' in part:
                    key, value = part.split('=', 1)
                    if key:
                        # Replace hyphens and underscores with spaces in keys
                        tokens.append(key.replace('-', ' ').replace('_', ' '))
                        if value:
                            # Replace hyphens and underscores with spaces in values
                            tokens.append(value.replace(
                                '-', ' ').replace('_', ' '))
                else:
                    # Handle query parameters without values (e.g., ?key2)
                    tokens.append(part.replace('-', ' ').replace('_', ' '))

    # Add fragment (hashtag)
    if parsed.fragment:
        # Replace hyphens and underscores with spaces in fragment
        tokens.append(parsed.fragment.replace('-', ' ').replace('_', ' '))

    return tokens


# Example
if __name__ == "__main__":
    all_urls = get_all_sitemap_urls("https://sitegiant.ph")
    print(f"\nFound {len(all_urls)} URLs:")
    for url in all_urls[:10]:  # just print first 10 for brevity
        print("-", url)
