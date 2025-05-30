from typing import List
from urllib.parse import urlparse, parse_qs, urlencode
from .stratified_sampler import StratifiedSampler


def preprocess_url(url: str) -> List[str]:
    """Convert a URL into a list of tokens for n-gram processing."""
    # Handle empty URL
    if not url:
        return []

    # Ensure URL has a scheme if none provided
    if not url.startswith(('http://', 'https://')):
        url = f'https://{url}'

    parsed = urlparse(url)
    tokens = []

    # Add scheme (e.g., http, https)
    if parsed.scheme:
        tokens.append(parsed.scheme)

    # Add domain components
    if parsed.netloc:
        domain_parts = parsed.netloc.split('.')
        tokens.extend([part for part in domain_parts if part])

    # Add path segments
    if parsed.path and parsed.path != '/':
        path_segments = parsed.path.strip('/').split('/')
        tokens.extend([segment for segment in path_segments if segment])

    # Add query parameters, preserving duplicates
    if parsed.query:
        # Split query string manually to preserve all values
        query_parts = parsed.query.split('&')
        for part in query_parts:
            if '=' in part:
                key, value = part.split('=', 1)
                if key:
                    tokens.append(key)
                    if value:
                        tokens.append(value)

    # Add fragment (hashtag)
    if parsed.fragment:
        tokens.append(parsed.fragment)

    return tokens


def sample_diverse_urls(urls: List[str], num_samples: int, n: int = 2, top_n: int = 2) -> List[str]:
    """Sample diverse URLs using StratifiedSampler."""
    if not urls:
        return []

    # Convert URLs to tokenized strings for processing
    tokenized_urls = [' '.join(preprocess_url(url)) for url in urls]
    sampler = StratifiedSampler(tokenized_urls, num_samples=num_samples)
    sampled_tokenized = sampler.filter_strings(n=n, top_n=top_n)

    # Map back to original URLs, preserving order
    result = []
    seen = set()
    for sampled in sampled_tokenized:
        idx = tokenized_urls.index(sampled)
        if urls[idx] not in seen:
            result.append(urls[idx])
            seen.add(urls[idx])

    return result[:num_samples]
