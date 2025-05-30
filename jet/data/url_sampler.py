from typing import List
from urllib.parse import urlparse, parse_qs, urlencode

from jet.utils.url_utils import parse_url
from .stratified_sampler import StratifiedSampler


def sample_diverse_urls(urls: List[str], num_samples: int, n: int = 2, top_n: int = 2) -> List[str]:
    """Sample diverse URLs using StratifiedSampler."""
    if not urls:
        return []

    # Convert URLs to tokenized strings for processing
    tokenized_urls = [' '.join(parse_url(url)) for url in urls]
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
