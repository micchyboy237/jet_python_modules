from typing import List, Tuple
from jet.llm.mlx.mlx_types import LLMModelType
from jet.llm.mlx.models import resolve_model
from jet.logger import logger
import mlx.core as mx
from mlx_lm.evaluate import MLXLM
from lm_eval.api.instance import Instance
from tqdm import tqdm


def compute_perplexity(model: MLXLM, texts: List[str]) -> List[float]:
    # Prepare requests as Instance objects
    requests = [Instance(
        request_type="loglikelihood_rolling",
        doc={"text": text},
        arguments=(text,),
        idx=i
    ) for i, text in enumerate(texts)]

    # Compute log-likelihoods
    log_likelihoods = model.loglikelihood_rolling(requests)

    # Convert to perplexity
    perplexities = [float(mx.exp(-mx.array(ll) / len(model._tokenize([text])[0])))
                    for ll, text in zip(log_likelihoods, texts)]
    return perplexities


def compute_confidence(model: MLXLM, context: str, continuation: str) -> Tuple[List[float], bool]:
    # Prepare request as Instance object
    request = Instance(
        request_type="loglikelihood",
        doc={"context": context, "continuation": continuation},
        arguments=(context, continuation),
        idx=0
    )

    # Compute log-likelihood and greedy flag
    (logprob, is_greedy), = model.loglikelihood([request])

    # Tokenize context and continuation
    prefix = model._tokenize([context])[0]
    full_sequence = model._tokenize([context + continuation])[0]
    continuation_tokens = full_sequence[len(prefix):]

    # Process prompt to get initial logprobs
    logprobs, cache = model._process_prompt(prefix)

    # Compute per-token probabilities for continuation
    confidences = []
    if continuation_tokens:  # Handle non-empty continuation
        confidences.append(float(mx.exp(logprobs[0, continuation_tokens[0]])))
        if len(continuation_tokens) > 1:
            # Score remaining tokens
            inputs = mx.array(continuation_tokens[1:])[None, :]
            scores, _, _ = model._score_fn(inputs, cache=cache)
            confidences.extend([float(mx.exp(score)) for score in scores[0]])

    return confidences, is_greedy


def compute_top_confident_words_and_sequences(model: MLXLM, context: str, max_tokens: int = 10, num_sequences: int = 5) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    # Tokenize context
    prefix = model._tokenize([context])[0]

    # Compute log-probabilities for the next token
    logprobs, _ = model._process_prompt(prefix)
    probs = mx.softmax(logprobs[0], axis=-1)

    # Get top 5 token indices and probabilities
    top_indices = mx.argsort(-probs, axis=-1)[:5]
    top_probs = probs[top_indices].tolist()

    # Convert token indices to words
    tokenizer = model.tokenizer
    top_words = []
    for idx, prob in zip(top_indices.tolist(), top_probs):
        token = tokenizer.decode([idx])
        if token.strip() and not token.startswith("##"):  # Skip subword markers
            top_words.append((token.strip(), prob))

    # Generate top sequences using greedy decoding
    sequences = []
    current_context = context
    for _ in range(num_sequences):
        request = Instance(
            request_type="generate_until",
            doc={"context": current_context},
            arguments=(current_context, {
                       "until": ["\n\n"], "max_gen_tokens": max_tokens}),
            idx=0
        )
        generated = model.generate_until([request])[0]
        if not generated:
            continue

        full_sequence = model._tokenize([current_context + generated])[0]
        continuation_tokens = full_sequence[len(prefix):]
        if not continuation_tokens:
            continue

        logprobs, cache = model._process_prompt(prefix)
        confidences = [float(mx.exp(logprobs[0, continuation_tokens[0]]))]
        if len(continuation_tokens) > 1:
            inputs = mx.array(continuation_tokens[1:])[None, :]
            scores, _, _ = model._score_fn(inputs, cache=cache)
            confidences.extend([float(mx.exp(score)) for score in scores[0]])

        avg_confidence = sum(confidences) / \
            len(confidences) if confidences else 0.0
        sequences.append((generated, avg_confidence))
        current_context = context + generated

    sequences = sorted(sequences, key=lambda x: x[1], reverse=True)[
        :num_sequences]

    return top_words[:5], sequences


def compute_token_k_index_and_probability(model: MLXLM, context: str, continuation: str, k: int = 5) -> List[Tuple[str, int, float]]:
    # Tokenize context and continuation
    prefix = model._tokenize([context])[0]
    full_sequence = model._tokenize([context + continuation])[0]
    continuation_tokens = full_sequence[len(prefix):]

    # Handle empty continuation or prefix
    if not continuation_tokens or not prefix:
        logger.warning(
            "Empty prefix or continuation tokens. Returning empty result.")
        return []

    # Initialize result list
    results = []

    # Process prompt to get initial logprobs and cache
    try:
        logprobs, cache = model._process_prompt(prefix)
        logger.info(f"Initial logprobs shape: {logprobs.shape}")
        if logprobs.ndim != 2 or logprobs.shape[1] == 0:
            logger.error(
                f"Invalid logprobs shape from _process_prompt: {logprobs.shape}")
            return []
    except Exception as e:
        logger.error(f"Error in _process_prompt: {e}")
        return []

    # Compute probabilities for all continuation tokens
    try:
        if len(continuation_tokens) > 0:
            inputs = mx.array(continuation_tokens)[
                None, :]  # Shape: [1, seq_len]
            scores, _, _ = model._score_fn(inputs, cache=cache)
            logger.info(f"Scores shape from _score_fn: {scores.shape}")
            if scores.ndim != 3:
                logger.warning(
                    f"Expected scores shape [1, seq_len, vocab_size], got {scores.shape}. Falling back to logprobs.")
                scores = logprobs[:, None, :]  # Reshape to [1, 1, vocab_size]
            # Shape: [1, seq_len, vocab_size]
            all_probs = mx.softmax(scores, axis=-1)
        else:
            # Shape: [1, 1, vocab_size]
            all_probs = mx.softmax(logprobs, axis=-1)[:, None, :]
        logger.info(f"all_probs shape: {all_probs.shape}")
    except Exception as e:
        logger.warning(
            f"Error in _score_fn: {e}. Falling back to initial logprobs.")
        # Shape: [1, 1, vocab_size]
        all_probs = mx.softmax(logprobs, axis=-1)[:, None, :]
        logger.info(f"Fallback all_probs shape: {all_probs.shape}")

    # Process each continuation token
    for i, token_id in enumerate(continuation_tokens):
        # Get probabilities for current token
        try:
            # Ensure we don't exceed the sequence length
            seq_idx = min(i, all_probs.shape[1] - 1)
            probs = all_probs[0, seq_idx]  # Shape: [vocab_size]
            logger.info(f"Probs shape at position {i}: {probs.shape}")
            if probs.ndim != 1 or probs.shape[0] == 0:
                logger.warning(
                    f"Invalid probs shape at position {i}: {probs.shape}. Skipping token.")
                continue
        except Exception as e:
            logger.warning(
                f"Error accessing probs at position {i}: {e}. Skipping token.")
            continue

        # Get top-k indices and probabilities
        try:
            top_k_indices = mx.argsort(-probs, axis=-1)[:k]
            top_k_probs = probs[top_k_indices].tolist()
        except Exception as e:
            logger.warning(
                f"Error in argsort at position {i}: {e}. Skipping token.")
            continue

        # Find k-index and probability of current token
        k_index = -1
        token_prob = float(
            probs[token_id]) if token_id < probs.shape[0] else 0.0
        for rank, idx in enumerate(top_k_indices.tolist()):
            if idx == token_id:
                k_index = rank
                break

        # Decode token
        token_str = model.tokenizer.decode([token_id]).strip()

        # Append result if token is valid
        if token_str and k_index >= 0:
            results.append((token_str, k_index, token_prob))
            logger.info(
                f"Token: {token_str}, k_index: {k_index}, prob: {token_prob}")

    return results


# Load model
model_name: LLMModelType = "qwen3-1.7b-4bit"
model = MLXLM(resolve_model(model_name), max_tokens=2048)

# Test data for perplexity
test_data = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is transforming technology."
]

# Compute perplexity
perplexities = compute_perplexity(model, test_data)
for text, perp in zip(test_data, perplexities):
    print(f"Text: {text}\nPerplexity: {perp:.2f}")

# Test data for confidence and top words/sequences
context = "The quick brown fox"
continuation = " jumps over the lazy dog."
confidences, is_greedy = compute_confidence(model, context, continuation)
print(f"\nContext: {context}")
print(f"Continuation: {continuation}")
print(f"Token confidences: {[f'{c:.3f}' for c in confidences]}")
print(f"Is greedy: {is_greedy}")

# Compute top 5 words and sequences
top_words, top_sequences = compute_top_confident_words_and_sequences(
    model, context)
print(f"\nTop 5 next words and their confidences:")
for word, prob in top_words:
    print(f"Word: {word}, Confidence: {prob:.3f}")
print(f"\nTop 5 sequences and their average confidences:")
for seq, conf in top_sequences:
    print(f"Sequence: {seq}, Average Confidence: {conf:.3f}")

# Compute k index and probability for each token in continuation
token_results = compute_token_k_index_and_probability(
    model, context, continuation, k=5)
print(f"\nContinuation tokens' k index and probability:")
for token, k_index, prob in token_results:
    print(f"Token: {token}, k-index: {k_index}, Probability: {prob:.3f}")
