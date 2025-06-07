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
    Format instruction and query.

    Args:
        task_description: Task instruction string
        query: Query string

    Returns:
        str: Formatted instruction-query string
    """
    return f'Instruct: {task_description}\nQuery:{query}'


def evaluate_relevance(
    queries,
    documents,
    task_description,
    model_name="mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
    max_length=1092
):
    """
    Evaluates the relevance of documents to queries using embedding similarity.

    Args:
        queries (list): List of query strings.
        documents (list): List of document strings.
        task_description (str): Task instruction for formatting queries.
        model_name (str): Name of the pre-trained model.
        max_length (int): Maximum sequence length for tokenization.

    Returns:
        list: List of lists containing similarity scores between queries and documents.
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    model, _ = load(model_name)

    # Format queries with task description
    formatted_queries = [get_detailed_instruct(
        task_description, q) for q in queries]
    input_texts = formatted_queries + documents

    # Tokenize inputs
    batch_dict = tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="np"
    )

    # Convert to MLX arrays
    input_ids = mx.array(batch_dict['input_ids'], dtype=mx.int32)
    attention_mask = mx.array(batch_dict['attention_mask'], dtype=mx.int32)

    # Compute embeddings
    outputs = model(input_ids)
    embeddings = last_token_pool(outputs, attention_mask)

    # Normalize embeddings
    embeddings = embeddings / \
        mx.sqrt(mx.sum(embeddings * embeddings, axis=1, keepdims=True))

    # Compute similarity scores
    query_embeddings = embeddings[:len(queries)]
    doc_embeddings = embeddings[len(queries):]
    scores = mx.matmul(query_embeddings, doc_embeddings.T)

    return scores.tolist()


# Example usage
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
    scores = evaluate_relevance(queries, documents, task, model_name)
    print("Scores:", scores)
