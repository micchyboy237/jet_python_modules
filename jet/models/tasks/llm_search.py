import uuid
from typing import List, TypedDict
import numpy as np
from llama_cpp import Llama
from sklearn.preprocessing import normalize
from tqdm import tqdm
from jet.models.tasks.task_types import SimilarityResult
from jet.models.tasks.utils import last_token_pool, get_detailed_instruct, encode_with_padding


def search_docs(
    model: Llama,
    queries: List[str],
    documents: List[str],
    task_description: str,
    max_length: int = 512,
    ids: List[str] = []
) -> List[List[SimilarityResult]]:
    """
    Search documents for relevance to queries using embeddings and return ranked results.

    Args:
        model: Llama model instance for embedding.
        queries: List of query strings.
        documents: List of document strings to search.
        task_description: Description of the task for query formatting.
        max_length: Maximum token length for encoding.
        ids: Optional list of IDs for documents. Must match length of documents if provided.

    Returns:
        List of lists of SimilarityResult, one list per query, sorted by score (descending).

    Raises:
        ValueError: If queries or documents are empty, or if ids length does not match documents.
        RuntimeError: If embedding or similarity computation fails.
    """
    if not queries or not documents:
        raise ValueError("Queries and documents must not be empty")
    if ids and len(ids) != len(documents):
        raise ValueError("Length of ids must match length of documents")

    try:
        # Format queries with task description
        formatted_queries = [get_detailed_instruct(
            task_description, q) for q in queries]

        # Encode queries and documents
        query_embeddings = encode_with_padding(
            model, formatted_queries, max_length)
        document_embeddings = encode_with_padding(model, documents, max_length)

        # Apply last-token pooling
        query_embeddings = last_token_pool(query_embeddings)
        document_embeddings = last_token_pool(document_embeddings)

        # Normalize embeddings
        query_embeddings = normalize(query_embeddings, norm='l2', axis=1)
        document_embeddings = normalize(document_embeddings, norm='l2', axis=1)

        # Compute similarity scores (queries vs documents)
        scores = query_embeddings @ document_embeddings.T

        # Process results for each query
        results: List[List[SimilarityResult]] = []
        for query_idx in range(len(queries)):
            query_scores = scores[query_idx]
            # Sort documents by score (descending)
            sorted_indices = np.argsort(query_scores)[::-1]
            query_results: List[SimilarityResult] = []
            for rank, doc_idx in enumerate(sorted_indices, 1):
                tokens = len(model.tokenize(
                    documents[doc_idx].encode('utf-8'), add_bos=True))
                result: SimilarityResult = {
                    'id': ids[doc_idx] if ids else str(uuid.uuid4()),
                    'rank': rank,
                    'doc_index': int(doc_idx),
                    'score': float(query_scores[doc_idx]),
                    'text': documents[doc_idx],
                    'tokens': tokens
                }
                query_results.append(result)
            results.append(query_results)
        return results

    except Exception as e:
        raise RuntimeError(f"Error during search: {str(e)}")
