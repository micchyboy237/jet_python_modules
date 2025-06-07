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
    max_length: int = 512
) -> List[List[SimilarityResult]]:
    """
    Search documents for relevance to queries using embeddings and return ranked results.

    Args:
        model: Llama model instance for embedding.
        queries: List of query strings.
        documents: List of document strings to search.
        task_description: Description of the task for query formatting.
        max_length: Maximum token length for encoding.

    Returns:
        List of lists of SimilarityResult, one list per query, sorted by score (descending).

    Raises:
        ValueError: If queries or documents are empty, or embeddings have inconsistent shapes.
        RuntimeError: If embedding or similarity computation fails.
    """
    if not queries or not documents:
        raise ValueError("Queries and documents must not be empty")

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
                    'id': str(uuid.uuid4()),
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


# Example usage and test
if __name__ == "__main__":
    model_path = "/Users/jethroestrada/Downloads/Qwen3-Embedding-0.6B-f16.gguf"
    model = Llama(
        model_path=model_path,
        embedding=True,
        n_ctx=2048,
        n_threads=4,
        n_gpu_layers=0,
        verbose=False
    )
    task = 'Given a web search query, retrieve relevant passages that answer the query'
    queries = [
        'What is the capital of China?',
        # 'Explain gravity'
    ]
    documents = [
        "The capital of China is Beijing.",
        "China is a country in East Asia with a rich history.",
        "Gravity is a force that attracts two bodies towards each other."
    ]

    try:
        results = search_docs(model, queries, documents, task)
        for query_idx, query_results in enumerate(results):
            print(f"\nQuery: {queries[query_idx]}")
            for res in query_results:
                print(
                    f"Rank: {res['rank']}, Score: {res['score']:.4f}, Text: {res['text']}")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        model.close()
