from typing import List, Optional

from jet.features.nlp_utils import get_word_counts_lemmatized
from .stratified_sampler import StratifiedSampler, ProcessedDataString


def sample_diverse_texts(
    texts: List[str],
    num_samples: Optional[int] = None,
    n: int = 2,
    top_n: int = 1,
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
    word_counts_lemmatized = get_word_counts_lemmatized(
        "\n\n".join(texts), min_count=2, as_score=False)

    if not texts:
        return []

    if not num_samples:
        num_samples = len(texts)

    # Tokenize each text (simple whitespace tokenizer here)
    tokenized_docs = [' '.join(doc.strip().split()) for doc in texts]

    # Create ProcessedDataString if category_values are provided
    if category_values:
        if len(category_values) != len(texts):
            raise ValueError(
                "Length of category_values must match length of texts")
        data = [
            ProcessedDataString(source=tokenized, category_values=cats)
            for tokenized, cats in zip(tokenized_docs, category_values)
        ]
    else:
        data = tokenized_docs

    # Initialize and use StratifiedSampler
    sampler = StratifiedSampler(data, num_samples=num_samples)
    sampled_tokenized = sampler.filter_strings(n=n, top_n=top_n)

    # Map back to original texts
    result = []
    seen = set()
    for sampled in sampled_tokenized:
        idx = tokenized_docs.index(sampled)
        if texts[idx] not in seen:
            result.append(texts[idx])
            seen.add(texts[idx])

    return result[:num_samples]
