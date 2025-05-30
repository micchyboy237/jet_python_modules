from typing import List
from urllib.parse import urlparse, parse_qs
from .stratified_sampler import StratifiedSampler


def preprocess_url(url: str) -> List[str]:
    """Convert a URL into a list of tokens for n-gram processing."""
    parsed = urlparse(url)
    tokens = []
    # Add scheme (e.g., http, https)
    tokens.append(parsed.scheme)
    # Add domain components
    domain_parts = parsed.netloc.split('.')
    tokens.extend(domain_parts)
    # Add path segments
    if parsed.path:
        path_segments = parsed.path.strip('/').split('/')
        tokens.extend(path_segments)
    # Add query parameters
    if parsed.query:
        query_params = parse_qs(parsed.query)
        for key, values in query_params.items():
            tokens.append(key)
            tokens.extend(values)
    return tokens


def sample_diverse_urls(urls: List[str], num_samples: int, n: int = 2, top_n: int = 2) -> List[str]:
    """Sample diverse URLs using StratifiedSampler."""
    # Convert URLs to tokenized strings for processing
    tokenized_urls = [' '.join(preprocess_url(url)) for url in urls]
    sampler = StratifiedSampler(tokenized_urls, num_samples=num_samples)
    sampled_tokenized = sampler.filter_strings(n=n, top_n=top_n)
    # Map back to original URLs
    return [urls[tokenized_urls.index(sampled)] for sampled in sampled_tokenized]
