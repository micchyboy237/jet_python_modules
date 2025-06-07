from tqdm import tqdm
from jet.logger import logger
from jet.models.embeddings.base import generate_embeddings
from jet.models.tokenizer.base import get_tokenizer
import mlx.core as mx
import numpy as np
from mlx_lm import load
from typing import Union, List
import math


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
    return f'Instruct: {task_description}\nQuery:{query}'


def evaluate_relevance(
    queries: Union[str, List[str]],
    documents: List[str],
    task_description: str,
    model_name: str = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
    max_length: Union[int, None] = None,
    batch_size: Union[int, None] = None,
    show_progress: bool = False,
    doc_batch_size: int = 50  # Reduced default to manage memory
) -> List[List[float]]:
    model, _ = load(model_name)
    tokenizer = get_tokenizer(model_name)
    queries = [queries] if isinstance(queries, str) else queries
    formatted_queries = [get_detailed_instruct(
        task_description, q) for q in queries]
    input_texts = formatted_queries + documents
    optimal_batch_size = batch_size if batch_size else 4  # Conservative default
    if show_progress:
        logger.debug(f"Optimal batch size: {optimal_batch_size}")

    input_ids_list: list[list[float]] = generate_embeddings(
        input_texts, model_name, optimal_batch_size, show_progress)
    optimal_max_length = max(len(input_ids) for input_ids in input_ids_list)
    if show_progress:
        logger.debug(f"Optimal max length: {optimal_max_length}")

    padded_inputs = []
    attention_masks = []
    for ids in tqdm(input_ids_list, desc="Padding Inputs", disable=not show_progress):
        if len(ids) > optimal_max_length:
            ids = ids[:optimal_max_length]
        padded = ids + [0] * (optimal_max_length - len(ids))
        mask = [1] * len(ids) + [0] * (optimal_max_length - len(ids))
        padded_inputs.append(padded)
        attention_masks.append(mask)

    if show_progress:
        logger.debug("Converting padded inputs to MLX array")
    input_ids = mx.array(
        np.array(padded_inputs, dtype=np.int32), dtype=mx.int32)
    if show_progress:
        logger.debug("Converting attention masks to MLX array")
    attention_mask = mx.array(
        np.array(attention_masks, dtype=np.int32), dtype=mx.int32)
    if show_progress:
        logger.debug("Running model inference")
    outputs = model(input_ids)
    if show_progress:
        logger.debug("Computing embeddings with last token pooling")
    embeddings = last_token_pool(outputs, attention_mask)
    embeddings = mx.concatenate([embeddings], axis=0)
    embeddings = embeddings / \
        mx.sqrt(mx.sum(embeddings * embeddings, axis=1, keepdims=True) + 1e-8)

    if show_progress:
        logger.debug("Separating query and document embeddings")
    query_embeddings = embeddings[:len(queries)]
    doc_embeddings = embeddings[len(queries):]

    scores = []
    num_query_batches = math.ceil(len(queries) / optimal_batch_size)
    num_doc_batches = math.ceil(len(doc_embeddings) / doc_batch_size)

    for i in tqdm(
        range(0, len(queries), optimal_batch_size),
        desc="Processing query batches",
        total=num_query_batches,
        disable=not show_progress
    ):
        batch_query_embeddings = query_embeddings[i:i + optimal_batch_size]
        batch_scores = []
        doc_progress = tqdm(
            range(0, len(doc_embeddings), doc_batch_size),
            desc="Processing document batches",
            total=num_doc_batches,
            leave=False,
            disable=not show_progress
        )
        for j in doc_progress:
            doc_batch = doc_embeddings[j:j + doc_batch_size]
            partial_scores = mx.matmul(batch_query_embeddings, doc_batch.T)
            # Convert incrementally
            batch_scores.append(partial_scores.tolist())
        batch_scores = np.concatenate(batch_scores, axis=1).tolist()
        scores.extend(batch_scores)

    return scores


if __name__ == "__main__":
    model_name = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
    task = 'Given a web search query, retrieve relevant passages that answer the query'
    queries = [
        "What is the capital of China?",
        "Explain gravity",
    ]
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ]
    scores = evaluate_relevance(
        queries, documents, task, model_name, show_progress=True)
    print("Scores:", scores)
