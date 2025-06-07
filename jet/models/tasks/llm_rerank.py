import uuid
from typing import List, TypedDict
import numpy as np
from llama_cpp import Llama
from sklearn.preprocessing import normalize
from tqdm import tqdm
from jet.models.tasks.task_types import SimilarityResult, RerankResult
from jet.models.tasks.utils import last_token_pool, get_detailed_instruct, encode_with_padding


def rerank_docs(
    model: Llama,
    queries: List[str],
    search_results: List[List[SimilarityResult]],
    max_length: int = 512
) -> List[List[RerankResult]]:
    """
    Rerank search results using a heuristic based on token overlap and length normalization.

    Args:
        model: Llama model instance for tokenization.
        queries: List of original query strings.
        search_results: List of lists of SimilarityResult from search_docs.
        max_length: Maximum token length for tokenization.

    Returns:
        List of lists of RerankResult, one list per query, sorted by reranked score (descending).

    Raises:
        ValueError: If queries or search_results are empty or mismatched in length.
        RuntimeError: If reranking computation fails.
    """
    if not queries or not search_results or len(queries) != len(search_results):
        raise ValueError(
            "Queries and search_results must be non-empty and have matching lengths")

    try:
        results: List[List[RerankResult]] = []
        for query, query_results in zip(queries, search_results):
            if not query_results:
                results.append([])
                continue

            # Tokenize query
            query_tokens = set(model.tokenize(
                query.encode('utf-8'), add_bos=True))

            # Compute reranked scores (heuristic: token overlap normalized by length)
            rerank_scores = []
            for res in query_results:
                doc_tokens = set(model.tokenize(
                    res['text'].encode('utf-8'), add_bos=True))
                overlap = len(query_tokens.intersection(doc_tokens))
                # Normalize by document length to favor concise, relevant docs
                normalized_score = overlap / \
                    max(1, len(doc_tokens)) * res['score']
                rerank_scores.append(normalized_score)

            # Sort by reranked scores
            sorted_indices = np.argsort(rerank_scores)[::-1]
            reranked_results: List[RerankResult] = []
            for rank, idx in enumerate(sorted_indices, 1):
                orig_result = query_results[idx]
                result: RerankResult = {
                    'id': orig_result['id'],
                    'rank': rank,
                    'doc_index': orig_result['doc_index'],
                    'score': float(rerank_scores[idx]),
                    'text': orig_result['text'],
                    'tokens': orig_result['tokens']
                }
                reranked_results.append(result)
            results.append(reranked_results)
        return results

    except Exception as e:
        raise RuntimeError(f"Error during reranking: {str(e)}")


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
        # Initial search
        search_results = search_docs(model, queries, documents, task)
        for query_idx, query_results in enumerate(search_results):
            print(f"\nQuery: {queries[query_idx]} (Search Results)")
            for res in query_results:
                print(
                    f"Rank: {res['rank']}, Score: {res['score']:.4f}, Text: {res['text']}")

        # Rerank results
        rerank_results = rerank_docs(model, queries, search_results)
        for query_idx, query_results in enumerate(rerank_results):
            print(f"\nQuery: {queries[query_idx]} (Reranked Results)")
            for res in query_results:
                print(
                    f"Rank: {res['rank']}, Score: {res['score']:.4f}, Text: {res['text']}")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        model.close()
