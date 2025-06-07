import uuid
from typing import List, Union, Literal
from typing_extensions import TypedDict
import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer
from mlx_lm import load
from jet.models.tasks.evaluate_relevance import evaluate_relevance


class SimilarityResult(TypedDict):
    """
    Represents a single similarity result for a text.

    Fields:
        id: Identifier for the text. (Use uuid if ids are not provided)
        rank: Rank based on score (1 for highest).
        doc_index: Original index of the text in the input list.
        score: Normalized similarity score.
        text: The compared text (or chunk if long).
        tokens: Number of tokens from text.
    """
    id: str
    rank: int
    doc_index: int
    score: float
    text: str
    tokens: int


def search_docs(
    queries: Union[str, List[str]],
    documents: List[str],
    task_description: str,
    model_name: str = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
    max_length: int = 8192,
    fuse_method: Literal["average", "max", "none"] = "average"
) -> Union[List[SimilarityResult], List[List[SimilarityResult]]]:
    """
    Searches documents for relevance to one or more queries and returns ranked results.

    Args:
        queries: Single query string or list of query strings.
        documents: List of document strings.
        task_description: Task instruction for formatting queries.
        model_name: Name of the pre-trained model.
        max_length: Maximum sequence length for tokenization.
        fuse_method: Method to fuse scores across multiple queries ("average", "max", or "none").
                     If "none", returns a list of lists, with results per query.

    Returns:
        List of SimilarityResult dictionaries if fuse_method is "average" or "max",
        or List of Lists of SimilarityResult dictionaries if fuse_method is "none".
    """
    # Convert single query to list for consistent processing
    if isinstance(queries, str):
        queries = [queries]

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')

    # Get similarity scores for all queries
    scores = evaluate_relevance(
        queries, documents, task_description, model_name, max_length)

    # Count tokens for each document
    tokenized_docs = [
        tokenizer(doc, truncation=True, max_length=max_length) for doc in documents]
    token_counts = [len(encoding['input_ids']) for encoding in tokenized_docs]

    if fuse_method == "none":
        # Return separate results for each query
        all_results = []
        for query_scores in scores:
            # Create list of tuples with (index, score, text, tokens) for this query
            results = [
                (i, score, documents[i], token_counts[i])
                for i, score in enumerate(query_scores)
            ]
            # Sort by score in descending order to assign ranks
            results.sort(key=lambda x: x[1], reverse=True)
            # Create SimilarityResult list for this query
            query_results = [
                {
                    'id': str(uuid.uuid4()),
                    'rank': rank + 1,
                    'doc_index': doc_index,
                    'score': float(score),  # Convert to Python float
                    'text': text,
                    'tokens': tokens
                }
                for rank, (doc_index, score, text, tokens) in enumerate(results)
            ]
            all_results.append(query_results)
        return all_results

    # Fuse scores based on fuse_method
    if fuse_method == "average":
        fused_scores = np.mean(scores, axis=0).tolist()
    elif fuse_method == "max":
        fused_scores = np.max(scores, axis=0).tolist()
    else:
        raise ValueError("fuse_method must be 'average', 'max', or 'none'")

    # Create list of tuples with (index, score, text, tokens)
    results = [
        (i, score, documents[i], token_counts[i])
        for i, score in enumerate(fused_scores)
    ]

    # Sort by score in descending order to assign ranks
    results.sort(key=lambda x: x[1], reverse=True)

    # Create SimilarityResult list
    similarity_results: List[SimilarityResult] = [
        {
            'id': str(uuid.uuid4()),
            'rank': rank + 1,
            'doc_index': doc_index,
            'score': float(score),  # Convert to Python float
            'text': text,
            'tokens': tokens
        }
        for rank, (doc_index, score, text, tokens) in enumerate(results)
    ]

    return similarity_results


# Example usage
if __name__ == "__main__":
    task = 'Given a web search query, retrieve relevant passages that answer the query'
    queries = [
        "What is the capital of China?",
        "Where is Beijing located?"
    ]
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other.",
        "Beijing is in northern China."
    ]

    # Example single query
    results_fused = search_docs(
        queries[0], documents, task, fuse_method="average")
    print("Single Query Results:")
    for result in results_fused:
        print(result)

    # Example with fusion
    results_fused = search_docs(
        queries, documents, task, fuse_method="average")
    print("Fused Results (average):")
    for result in results_fused:
        print(result)

    results_fused = search_docs(queries, documents, task, fuse_method="max")
    print("Fused Results (max):")
    for result in results_fused:
        print(result)

    # Example without fusion
    results_non_fused = search_docs(
        queries, documents, task, fuse_method="none")
    print("\nNon-Fused Results:")
    for i, query_results in enumerate(results_non_fused):
        print(f"\nQuery {i+1}: {queries[i]}")
        for result in query_results:
            print(result)
