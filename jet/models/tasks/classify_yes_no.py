import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer
from mlx_lm import load


def evaluate_relevance(
    queries,
    documents,
    instruction=None,
    model_name="mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
    max_length=1024,
    prefix_str="<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n",
    suffix_str="<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
):
    """
    Evaluates the relevance of documents to queries using a pre-trained model.

    Args:
        queries (list): List of query strings.
        documents (list): List of document strings.
        instruction (str, optional): Custom instruction for relevance judgment.
        model_name (str): Name of the pre-trained model.
        max_length (int): Maximum sequence length for tokenization.
        prefix_str (str): Prefix string for input formatting.
        suffix_str (str): Suffix string for input formatting.

    Returns:
        list: List of relevance scores (probabilities of "yes").
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    model, _ = load(model_name)

    # Get token IDs for "yes" and "no"
    token_false_id = tokenizer.convert_tokens_to_ids("no")
    token_true_id = tokenizer.convert_tokens_to_ids("yes")

    # Encode prefix and suffix
    prefix_tokens = list(tokenizer.encode(
        prefix_str, padding=True, add_special_tokens=False))
    suffix_tokens = list(tokenizer.encode(
        suffix_str, padding=True, add_special_tokens=False))
    num_reserved = len(prefix_tokens) + len(suffix_tokens)

    # Format input pairs
    default_instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    pairs = [
        "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
            instruction=instruction if instruction else default_instruction,
            query=query,
            doc=doc
        )
        for query, doc in zip(queries, documents)
    ]

    # Process inputs
    inputs = tokenizer(
        pairs,
        padding=False,
        truncation=True,
        max_length=max_length - num_reserved,
        return_tensors="np"
    )
    input_ids_list = inputs['input_ids'].tolist()

    input_ids = []
    for single_input in input_ids_list:
        combined = prefix_tokens + list(single_input) + suffix_tokens
        if len(combined) > max_length:
            max_input_len = max_length - \
                len(prefix_tokens) - len(suffix_tokens)
            truncated_input = single_input[:max_input_len]
            combined = prefix_tokens + truncated_input + suffix_tokens
        input_ids.append(combined)

    padded_inputs = tokenizer.pad(
        {'input_ids': input_ids},
        padding='max_length',
        max_length=max_length,
        return_tensors="np"
    )

    input_ids = mx.array(padded_inputs['input_ids'], dtype=mx.int32)
    attention_mask = mx.array(padded_inputs['attention_mask'], dtype=mx.int32)
    inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}

    # Compute logits
    logits = model(inputs['input_ids'])
    batch_scores = logits[:, -1, :]  # Shape: [batch_size, vocab_size]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    stacked = mx.stack([false_vector, true_vector], axis=1)
    probs = mx.softmax(stacked, axis=1)
    scores = probs[:, 1].tolist()

    return scores


# Example usage
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
    scores = evaluate_relevance(queries, documents, instruction=task)
    print("Scores:", scores)
