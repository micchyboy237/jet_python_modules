import mlx.core as mx
import numpy as np
from jet.models.embeddings.base import get_embedding_function, calculate_batch_size
from mlx_lm import load
from typing import Union, List


def last_token_pool(last_hidden_states: mx.array, attention_mask: mx.array) -> mx.array:
    left_padding = mx.sum(attention_mask[:, -1]) == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = mx.sum(attention_mask, axis=1) - 1
        batch_size = last_hidden_states.shape[0]
        indices = mx.stack([mx.arange(batch_size), sequence_lengths], axis=1)
        return last_hidden_states[indices[:, 0], indices[:, 1]]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery:{query}"


def compute_similarity(query_embeddings: mx.array, doc_embeddings: mx.array, normalize: bool = True) -> list:
    if normalize:
        query_embeddings = query_embeddings / \
            mx.sqrt(mx.sum(query_embeddings * query_embeddings,
                    axis=1, keepdims=True) + 1e-8)
        doc_embeddings = doc_embeddings / \
            mx.sqrt(mx.sum(doc_embeddings * doc_embeddings,
                    axis=1, keepdims=True) + 1e-8)
    return mx.matmul(query_embeddings, doc_embeddings.T).tolist()


def evaluate_similarity(
    queries: Union[str, List[str]],
    documents: List[str],
    task_description: str,
    model_name: str = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
    max_length: int = 512,
    batch_size: Union[int, None] = None,
    show_progress: bool = False,
    normalize: bool = True
) -> List[List[float]]:
    """Evaluate similarity between queries and documents using batched embeddings.

    Args:
        queries: Single query or list of queries.
        documents: List of documents to evaluate.
        task_description: Description of the task for query formatting.
        model_name: Name of the pre-trained model.
        max_length: Maximum sequence length for tokenization.
        batch_size: Batch size for tokenization and inference. If None, calculated dynamically.
        show_progress: Whether to show a progress bar.
        normalize: Whether to normalize embeddings before computing similarity.

    Returns:
        List of similarity scores for each query-document pair.
    """
    if not queries or not documents:
        return [[]]

    queries = [queries] if isinstance(queries, str) else queries
    model, _ = load(model_name)
    embed_func = get_embedding_function(model_name, batch_size, show_progress)

    # Format queries and combine with documents
    formatted_queries = [get_detailed_instruct(
        task_description, q) for q in queries]
    input_texts = formatted_queries + documents

    # Get token IDs for all texts using embed_func
    input_ids_list = embed_func(input_texts)

    # Ensure input_ids_list contains lists
    if isinstance(input_ids_list, float) or (isinstance(input_ids_list, list) and isinstance(input_ids_list[0], float)):
        input_ids_list = [input_ids_list] if isinstance(
            input_ids_list, float) else input_ids_list

    # Calculate batch size for inference
    optimal_batch_size = calculate_batch_size(input_texts, batch_size)

    # Process texts in batches for model inference
    embeddings = []
    for i in range(0, len(input_texts), optimal_batch_size):
        batch_ids = input_ids_list[i:i + optimal_batch_size]

        # Pad and create attention masks
        padded_inputs = []
        attention_masks = []
        for ids in batch_ids:
            if not isinstance(ids, list):
                ids = [ids]  # Handle single float case
            if len(ids) > max_length:
                ids = ids[:max_length]
            padded = ids + [0] * (max_length - len(ids))
            mask = [1] * len(ids) + [0] * (max_length - len(ids))
            padded_inputs.append(padded)
            attention_masks.append(mask)

        # Converting to mx.array
        input_ids = mx.array(
            np.array(padded_inputs, dtype=np.int32), dtype=mx.int32)
        attention_mask = mx.array(
            np.array(attention_masks, dtype=np.int32), dtype=mx.int32)

        # Model inference
        outputs = model(input_ids)
        batch_embeddings = last_token_pool(outputs, attention_mask)
        embeddings.append(batch_embeddings)

    embeddings = mx.concatenate(embeddings, axis=0)
    embeddings = embeddings / \
        mx.sqrt(mx.sum(embeddings * embeddings, axis=1, keepdims=True) + 1e-8)

    # Split embeddings
    query_embeddings = embeddings[:len(queries)]
    doc_embeddings = embeddings[len(queries):]

    # Compute similarity scores in batches
    scores = []
    for i in range(0, len(queries), optimal_batch_size):
        batch_query_embeddings = query_embeddings[i:i + optimal_batch_size]
        batch_scores = compute_similarity(
            batch_query_embeddings, doc_embeddings, normalize=normalize)
        scores.extend(batch_scores)

    return scores


if __name__ == "__main__":
    model_name = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
    task = "Given a web search query, retrieve relevant passages that answer the query"
    queries = [
        "What is the capital of China?",
        "Explain gravity",
    ]
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ]
    similarity_normalized = evaluate_similarity(
        queries, documents, task, model_name, show_progress=True)
    print("Normalized similarity scores:", similarity_normalized)
    similarity_raw = evaluate_similarity(
        queries, documents, task, model_name, normalize=False, show_progress=True)
    print("Raw similarity scores:", similarity_raw)
