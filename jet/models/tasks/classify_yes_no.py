import mlx.core as mx
import numpy as np
from jet.models.embeddings.base import get_embedding_function
from transformers import AutoTokenizer
from mlx_lm import load
from typing import Union, List


def evaluate_relevance(
    queries: Union[str, List[str]],
    documents: List[str],
    instruction: Union[str, None] = None,
    model_name: str = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
    max_length: int = 512,
    prefix_str: str = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n",
    suffix_str: str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    batch_size: Union[int, None] = None,
    show_progress: bool = False
) -> List[float]:
    """Evaluate yes/no relevance of documents to queries using batched embeddings.

    Args:
        queries: Single query or list of queries.
        documents: List of documents to evaluate.
        instruction: Optional task instruction. Defaults to web search query relevance.
        model_name: Name of the pre-trained model.
        max_length: Maximum sequence length for tokenization.
        prefix_str: Prefix string for input formatting.
        suffix_str: Suffix string for input formatting.
        batch_size: Batch size for tokenization. If None, calculated dynamically.
        show_progress: Whether to show a progress bar.

    Returns:
        List of relevance scores (probability of "yes") for each query-document pair.
    """
    queries = [queries] if isinstance(queries, str) else queries
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    embed_func = get_embedding_function(model_name, batch_size, show_progress)
    model, _ = load(model_name)

    # Get token IDs for prefix and suffix using AutoTokenizer
    prefix_ids = tokenizer.encode(prefix_str, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix_str, add_special_tokens=False)
    num_reserved = len(prefix_ids) + len(suffix_ids)

    # Get token IDs for "yes" and "no" using AutoTokenizer
    token_false_id = tokenizer.convert_tokens_to_ids("no")
    token_true_id = tokenizer.convert_tokens_to_ids("yes")

    default_instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    pairs = [
        "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
            instruction=instruction if instruction else default_instruction,
            query=query,
            doc=doc
        )
        for query, doc in zip(queries, documents)
    ]

    # Tokenize pairs using get_embedding_function
    input_ids_list = embed_func(pairs)

    # Combine with prefix/suffix and pad
    input_ids = []
    attention_masks = []
    for single_input in input_ids_list:
        combined = prefix_ids + single_input + suffix_ids
        if len(combined) > max_length:
            max_input_len = max_length - len(prefix_ids) - len(suffix_ids)
            combined = prefix_ids + single_input[:max_input_len] + suffix_ids
        padded = combined + [0] * (max_length - len(combined))
        mask = [1] * len(combined) + [0] * (max_length - len(combined))
        input_ids.append(padded)
        attention_masks.append(mask)

    # Convert to mx.array
    input_ids = mx.array(np.array(input_ids, dtype=np.int32), dtype=mx.int32)
    attention_mask = mx.array(
        np.array(attention_masks, dtype=np.int32), dtype=mx.int32)

    # Model inference
    logits = model(input_ids)
    batch_scores = logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    stacked = mx.stack([false_vector, true_vector], axis=1)
    probs = mx.softmax(stacked, axis=1)
    scores = probs[:, 1].tolist()

    return scores


if __name__ == "__main__":
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
        queries, documents, instruction=task, batch_size=32)
    print("Scores:", scores)
