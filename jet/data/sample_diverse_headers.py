from typing import List, Optional
from jet.vectors.document_types import HeaderDocument
from .stratified_sampler import StratifiedSampler, ProcessedDataString


def sample_diverse_headers(
    docs: List[HeaderDocument],
    num_samples: Optional[int] = None,
    n: int = 2,
    top_n: int = 1,
    category_values: Optional[List[List[str]]] = None
) -> List[HeaderDocument]:
    """Sample diverse documents using StratifiedSampler.

    Args:
        docs: List of HeaderDocument objects to sample from.
        num_samples: Number of documents to sample. If None, uses all documents.
        n: N-gram size for diversity sampling.
        top_n: Number of documents to keep per n-gram group.
        category_values: Optional list of category value lists for stratification.

    Returns:
        List of sampled HeaderDocument objects.
    """
    if not docs:
        return []

    if not num_samples:
        num_samples = len(docs)

    # Extract text content from documents
    texts = [doc.text for doc in docs]

    # Tokenize each text (simple whitespace tokenizer here)
    tokenized_docs = [' '.join(doc.strip().split()) for doc in texts]

    # Create ProcessedDataString if category_values are provided
    if category_values:
        if len(category_values) != len(docs):
            raise ValueError(
                "Length of category_values must match length of docs")
        data = [
            ProcessedDataString(source=tokenized, category_values=cats)
            for tokenized, cats in zip(tokenized_docs, category_values)
        ]
    else:
        data = tokenized_docs

    # Initialize and use StratifiedSampler
    sampler = StratifiedSampler(data, num_samples=num_samples)
    sampled_tokenized = sampler.filter_strings(n=n, top_n=top_n)

    # Map back to original documents
    result = []
    seen = set()
    for sampled in sampled_tokenized:
        idx = tokenized_docs.index(sampled)
        if docs[idx] not in seen:
            result.append(docs[idx])
            seen.add(docs[idx])

    return result
