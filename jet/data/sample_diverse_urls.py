from typing import List, Optional

from .stratified_sampler import StratifiedSampler, ProcessedDataString


def sample_diverse_urls(
    urls: List[str],
    num_samples: Optional[int] = None,
    n: int = 2,
    top_n: int = 1,
    category_values: Optional[List[List[str]]] = None
) -> List[str]:
    """Sample diverse URLs using StratifiedSampler.

    Args:
        urls: List of URLs to sample from.
        num_samples: Number of URLs to sample. If None, uses all URLs.
        n: N-gram size for diversity sampling.
        top_n: Number of sentences to keep per n-gram group.
        category_values: Optional list of category value lists for stratification.

    Returns:
        List of sampled URLs.
    """
    from jet.utils.url_utils import clean_url, parse_url

    if not urls:
        return []

    if not num_samples:
        num_samples = len(urls)

    # Prepare data for StratifiedSampler
    tokenized_urls = [' '.join(parse_url(clean_url(url))) for url in urls]

    # Create ProcessedDataString if category_values are provided
    if category_values:
        if len(category_values) != len(urls):
            raise ValueError(
                "Length of category_values must match length of urls")
        data = [
            ProcessedDataString(source=tokenized, category_values=cats)
            for tokenized, cats in zip(tokenized_urls, category_values)
        ]
    else:
        data = tokenized_urls

    # Initialize and use StratifiedSampler
    sampler = StratifiedSampler(data, num_samples=num_samples)
    sampled_tokenized = sampler.filter_strings(n=n, top_n=top_n)

    # Map back to original URLs
    result = []
    seen = set()
    for sampled in sampled_tokenized:
        idx = tokenized_urls.index(sampled)
        if urls[idx] not in seen:
            result.append(urls[idx])
            seen.add(urls[idx])

    return result[:num_samples]
