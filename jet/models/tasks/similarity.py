import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer
from mlx_lm import load


def last_token_pool(last_hidden_states: mx.array, attention_mask: mx.array) -> mx.array:
    """
    Pool the last token's hidden state, accounting for padding.

    Args:
        last_hidden_states: Shape [batch_size, seq_len, hidden_size]
        attention_mask: Shape [batch_size, seq_len]

    Returns:
        mx.array: Pooled embeddings, shape [batch_size, hidden_size]
    """
    left_padding = mx.sum(attention_mask[:, -1]) == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = mx.sum(attention_mask, axis=1) - 1
        batch_size = last_hidden_states.shape[0]
        indices = mx.stack([mx.arange(batch_size), sequence_lengths], axis=1)
        return last_hidden_states[indices[:, 0], indices[:, 1]]


def get_detailed_instruct(task_description: str, query: str) -> str:
    """
    Format a query with a task description, matching the non-MLX version.

    Args:
        task_description: Task instruction string
        query: Query string

    Returns:
        str: Formatted query string
    """
    return f"Instruct: {task_description}\nQuery:{query}"


def encode_texts(texts, tokenizer, model, max_length=8192, batch_size=32):
    """
    Encode a list of texts into embeddings using the MLX model.

    Args:
        texts: List of text strings to encode
        tokenizer: Hugging Face tokenizer
        model: MLX model
        max_length: Maximum sequence length
        batch_size: Batch size for encoding

    Returns:
        mx.array: Embeddings, shape [num_texts, hidden_size]
    """
    if not texts:
        return mx.array([], dtype=mx.float32)

    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_dict = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="np"
        )
        input_ids = mx.array(batch_dict['input_ids'], dtype=mx.int32)
        attention_mask = mx.array(batch_dict['attention_mask'], dtype=mx.int32)
        outputs = model(input_ids)
        batch_embeddings = last_token_pool(outputs, attention_mask)
        embeddings.append(batch_embeddings)
    return mx.concatenate(embeddings, axis=0)


def compute_similarity(query_embeddings: mx.array, doc_embeddings: mx.array, normalize: bool = True) -> list:
    """
    Compute similarity between query and document embeddings.

    Args:
        query_embeddings: Shape [num_queries, hidden_size]
        doc_embeddings: Shape [num_docs, hidden_size]
        normalize: If True, normalize embeddings for cosine similarity; if False, use raw dot product

    Returns:
        list: Similarity matrix, shape [num_queries, num_docs]
    """
    if normalize:
        query_embeddings = query_embeddings / \
            mx.sqrt(mx.sum(query_embeddings * query_embeddings,
                    axis=1, keepdims=True) + 1e-8)
        doc_embeddings = doc_embeddings / \
            mx.sqrt(mx.sum(doc_embeddings * doc_embeddings,
                    axis=1, keepdims=True) + 1e-8)
    return mx.matmul(query_embeddings, doc_embeddings.T).tolist()


def evaluate_similarity(
    queries,
    documents,
    task_description,
    model_name="mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
    max_length=8192,
    batch_size=32,
    normalize=True
):
    """
    Evaluates similarity between queries and documents using the MLX-quantized model.

    Args:
        queries (list): List of query strings
        documents (list): List of document strings
        task_description (str): Task instruction for query formatting
        model_name (str): Name of the MLX-quantized model
        max_length (int): Maximum sequence length for tokenization
        batch_size (int): Batch size for encoding
        normalize (bool): If True, normalize embeddings for cosine similarity

    Returns:
        list: Similarity matrix [num_queries, num_docs]
    """
    # Input validation
    if not queries or not documents:
        return [[]]

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    model, _ = load(model_name)

    # Format queries with task description
    formatted_queries = [get_detailed_instruct(
        task_description, q) for q in queries]
    input_texts = formatted_queries + documents

    # Encode all texts
    embeddings = encode_texts(input_texts, tokenizer,
                              model, max_length, batch_size)

    # Split embeddings into queries and documents
    query_embeddings = embeddings[:len(queries)]
    doc_embeddings = embeddings[len(queries):]

    # Compute similarity
    similarity = compute_similarity(
        query_embeddings, doc_embeddings, normalize=normalize)
    return similarity


# Example usage
if __name__ == "__main__":
    model_name = "mlx-community/Qwen3-Embedding-4B-4bit-DWQ"
    task = "Given a web search query, retrieve relevant passages that answer the query"
    queries = [
        "What is the capital of China?",
        "Explain gravity",
    ]
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ]
    # With normalization (matches PyTorch behavior)
    similarity_normalized = evaluate_similarity(
        queries, documents, task, model_name)
    print("Normalized similarity scores:", similarity_normalized)

    # Without normalization (for comparison)
    similarity_raw = evaluate_similarity(
        queries, documents, task, normalize=False)
    print("Raw similarity scores:", similarity_raw)
