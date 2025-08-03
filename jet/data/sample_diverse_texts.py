from typing import List, Optional

from jet.features.nlp_utils import get_word_counts_lemmatized
from .stratified_sampler import StratifiedSampler, ProcessedDataString


def sample_diverse_texts(
    texts: List[str],
    num_samples: Optional[int] = None,
    n: int = 2,
    top_n: Optional[int] = None,
    category_values: Optional[List[List[str]]] = None
) -> List[str]:
    """Sample diverse texts using StratifiedSampler.

    Args:
        texts: List of texts to sample from.
        num_samples: Number of texts to sample. If None, uses all texts.
        n: N-gram size for diversity sampling.
        top_n: Number of strings to keep per n-gram group.
        category_values: Optional list of category value lists for stratification.

    Returns:
        List of sampled texts.
    """

    # Initialize and use StratifiedSampler
    sampler = StratifiedSampler(texts, num_samples=num_samples)
    results = sampler.filter_strings(n=n, top_n=top_n)

    return results
