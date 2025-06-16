import torch
from typing import List, Optional
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel
import uuid
from tqdm import tqdm


class RerankResult(BaseModel):
    """
    Represents a single reranked result for a text.

    Fields:
        id: Identifier for the text
        rank: Updated rank based on reranked score (1 for highest).
        doc_index: Original index of the text in the input list.
        score: Reranked score
        text: The compared text (or chunk if long).
        tokens: Number of tokens from text.
    """
    id: str
    rank: int
    doc_index: int
    score: float
    text: str
    tokens: int


def format_instruction(instruction: Optional[str], query: str, doc: str) -> str:
    """
    Format the instruction, query, and document into a single string.

    Args:
        instruction: Optional instruction for the task.
        query: The search query.
        doc: The document to evaluate.

    Returns:
        Formatted string combining instruction, query, and document.
    """
    if instruction is None:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction, query=query, doc=doc)


def process_inputs(pairs: List[str], tokenizer: AutoTokenizer, prefix_tokens: List[int],
                   suffix_tokens: List[int], max_length: int, device: torch.device) -> dict:
    """
    Process input pairs into tokenized format for the model.

    Args:
        pairs: List of formatted input strings.
        tokenizer: Pre-trained tokenizer.
        prefix_tokens: Tokens for prefix.
        suffix_tokens: Tokens for suffix.
        max_length: Maximum length for tokenization.
        device: Device to move tensors to.

    Returns:
        Dictionary of tokenized inputs.
    """
    inputs = tokenizer(
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
    inputs = tokenizer.pad(inputs, padding=True,
                           return_tensors="pt", max_length=max_length)
    for key in inputs:
        inputs[key] = inputs[key].to(device)
    return inputs


@torch.no_grad()
def compute_logits(inputs: dict, model: AutoModelForCausalLM, token_true_id: int,
                   token_false_id: int) -> List[float]:
    """
    Compute relevance scores from model logits.

    Args:
        inputs: Tokenized inputs.
        model: Pre-trained model for reranking.
        token_true_id: Token ID for "yes".
        token_false_id: Token ID for "no".

    Returns:
        List of relevance scores.
    """
    batch_scores = model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    return batch_scores[:, 1].exp().tolist()


def rerank_docs(
    query: str,
    documents: List[str],
    instruction: Optional[str] = None,
    ids: Optional[List[str]] = None,
    model_name: str = "Qwen/Qwen3-Reranker-0.6B",
    max_length: int = 128,
    show_progress: bool = False,
    batch_size: int = 16
) -> List[RerankResult]:
    """
    Rerank documents based on their relevance to a single query using a transformer model.

    Args:
        query: Single search query.
        documents: List of documents to rerank.
        ids: Optional list of unique IDs for each document. Must match documents length if provided.
        instruction: Optional instruction for the task.
        model_name: Name of the pre-trained model to use.
        max_length: Maximum token length for input processing.
        show_progress: If True, display progress bars for processing steps.
        batch_size: Number of documents to process in each batch.

    Returns:
        List of RerankResult objects sorted by score in descending order.

    Raises:
        ValueError: If ids is provided and its length does not match documents length.
    """
    # Validate ids if provided
    if ids is not None and len(ids) != len(documents):
        raise ValueError("Length of ids must match length of documents")

    # Generate UUIDs if ids not provided
    doc_ids = ids if ids is not None else [
        str(uuid.uuid4()) for _ in documents]

    # Initialize device
    device = torch.device(
        "cpu")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()

    # Get token IDs for "yes" and "no"
    token_false_id = tokenizer.convert_tokens_to_ids("no")
    token_true_id = tokenizer.convert_tokens_to_ids("yes")

    # Prepare prefix and suffix tokens
    prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

    # Format input pairs
    pairs = [
        format_instruction(instruction, query, doc)
        for doc in tqdm(documents, desc="Formatting input pairs", disable=not show_progress)
    ]

    # Process documents in batches to reduce memory usage
    scores = []
    for i in tqdm(range(0, len(pairs), batch_size), desc="Processing batches", disable=not show_progress):
        batch_pairs = pairs[i:i + batch_size]
        inputs = process_inputs(
            batch_pairs, tokenizer, prefix_tokens, suffix_tokens, max_length, device)
        batch_scores = compute_logits(
            inputs, model, token_true_id, token_false_id)
        scores.extend(batch_scores)
        # Clear memory
        del inputs
        torch.mps.empty_cache() if device.type == "mps" else torch.cuda.empty_cache()

    # Count tokens for each document
    token_counts = [
        len(tokenizer.encode(doc, add_special_tokens=False))
        for doc in tqdm(documents, desc="Counting tokens", disable=not show_progress)
    ]

    # Create RerankResult objects
    results = [
        RerankResult(
            id=doc_id,
            rank=0,  # Will be updated after sorting
            doc_index=idx,
            score=score,
            text=doc,
            tokens=token_count
        )
        for idx, (doc_id, score, doc, token_count) in tqdm(
            enumerate(zip(doc_ids, scores, documents, token_counts)),
            total=len(documents),
            desc="Creating rerank results",
            disable=not show_progress
        )
    ]

    # Sort by score in descending order and assign ranks
    results.sort(key=lambda x: x.score, reverse=True)
    for rank, result in enumerate(results, 1):
        result.rank = rank

    return results


# Example usage and tests
if __name__ == "__main__":
    task = 'Given a web search query, retrieve relevant passages that answer the query'
    query = "What is the capital of China?"
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other."
    ]

    # Example with provided ids and progress bar
    results_with_ids = rerank_docs(
        query, documents, ids=["doc1", "doc2"], instruction=task, show_progress=True)
    for result in results_with_ids:
        print(result.dict())

    # Example without ids and with progress bar
    results_without_ids = rerank_docs(
        query, documents, instruction=task, show_progress=True)
    for result in results_without_ids:
        print(result.dict())
