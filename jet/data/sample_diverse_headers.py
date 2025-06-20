from typing import List, Optional, Union
from jet.vectors.document_types import HeaderDocument, HeaderDocumentWithScore
from jet.wordnet.words import get_words
from jet.data.stratified_sampler import StratifiedSampler, ProcessedDataString


def sample_diverse_headers(
    docs: Union[List[HeaderDocument], List[HeaderDocumentWithScore]],
    num_samples: Optional[int] = None,
    n: int = 2,
    top_n: int = 1,
    category_values: Optional[List[List[str]]] = None
) -> Union[List[HeaderDocument], List[HeaderDocumentWithScore]]:
    """Sample diverse documents using StratifiedSampler.

    Args:
        docs: List of HeaderDocument or HeaderDocumentWithScore objects to sample from.
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
    tokenized_docs = [doc.text for doc in docs]

    data: list[ProcessedDataString] = [
        ProcessedDataString(
            source=doc.text,
            category_values=[
                doc["source_url"],
                # (doc["parent_header"] or "").lstrip().strip().lower(),
                doc["header"].lstrip().strip().lower(),
            ]
        ) for doc in docs
    ]

    sampler = StratifiedSampler(data)
    # samples, _, _ = sampler.split_train_test_val(
    #     train_ratio=0.8, test_ratio=0.1)
    samples = sampler.get_samples()
    sampled_tokenized = [sample["source"] for sample in samples]

    # Map back to original documents
    result = []
    seen = set()
    for sampled in sampled_tokenized:
        idx = tokenized_docs.index(sampled)
        doc_id = docs[idx].id
        if doc_id not in seen:
            result.append(docs[idx])
            seen.add(doc_id)

    return result
