from typing import List, Tuple
from jet.models.model_types import LLMModelType
from jet.llm.mlx.models import resolve_model
from jet.logger import logger
import mlx.core as mx
from mlx_lm.evaluate import MLXLM
from lm_eval.api.instance import Instance
from tqdm import tqdm


def compute_perplexity(model: MLXLM, texts: List[str]) -> List[float]:
    requests = [Instance(
        request_type="loglikelihood_rolling",
        doc={"text": text},
        arguments=(text,),
        idx=i
    ) for i, text in enumerate(texts)]
    log_likelihoods = model.loglikelihood_rolling(requests)
    perplexities = [float(mx.exp(-mx.array(ll) / len(model._tokenize([text])[0])))
                    for ll, text in zip(log_likelihoods, texts)]
    return perplexities


def compute_confidence(model: MLXLM, context: str, continuation: str) -> Tuple[List[float], bool]:
    request = Instance(
        request_type="loglikelihood",
        doc={"context": context, "continuation": continuation},
        arguments=(context, continuation),
        idx=0
    )
    (logprob, is_greedy), = model.loglikelihood([request])
    prefix = model._tokenize([context])[0]
    full_sequence = model._tokenize([context + continuation])[0]
    continuation_tokens = full_sequence[len(prefix):]
    logprobs, cache = model._process_prompt(prefix)
    confidences = []
    if continuation_tokens:
        confidences.append(float(mx.exp(logprobs[0, continuation_tokens[0]])))
        if len(continuation_tokens) > 1:
            inputs = mx.array(continuation_tokens[1:])[None, :]
            scores, _, _ = model._score_fn(inputs, cache=cache)
            confidences.extend([float(mx.exp(score)) for score in scores[0]])
    return confidences, is_greedy


def compute_top_confident_words(model: MLXLM, context: str, top_k=10) -> List[Tuple[int, str, float]]:
    prefix = model._tokenize([context])[0]
    logprobs, _ = model._process_prompt(prefix)
    probs = mx.softmax(logprobs[0], axis=-1)
    top_indices = mx.argsort(-probs, axis=-1)[:top_k]
    top_probs = probs[top_indices].tolist()
    tokenizer = model.tokenizer
    top_words = []
    for k, (idx, prob) in enumerate(zip(top_indices.tolist(), top_probs)):
        token = tokenizer.decode([idx])
        if token.strip() and not token.startswith("##"):
            top_words.append((k, token.strip(), prob))
    return top_words  # [:top_k]


def compute_top_sequences(model: MLXLM, context: str, max_tokens: int = 10, num_sequences: int = 10) -> List[Tuple[str, str, float]]:
    prefix = model._tokenize([context])[0]
    top_words = compute_top_confident_words(model, context)
    sequences = []
    for _, start_word, _ in top_words:
        current_context = context + " " + start_word
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
        sequences.append((start_word, start_word + " " +
                         generated.strip(), avg_confidence))
    return sorted(sequences, key=lambda x: x[2], reverse=True)[:num_sequences]


def compute_token_k_index_and_probability(model: MLXLM, context: str, continuation: str, k: int = 10) -> List[Tuple[str, int, float]]:
    prefix = tuple(model._tokenize([context])[0])
    full_sequence = tuple(model._tokenize([context + continuation])[0])
    continuation_tokens = full_sequence[len(prefix):]
    if not continuation_tokens or not prefix:
        logger.warning(
            "Empty prefix or continuation tokens. Returning empty result.")
        return []
    results = []
    current_prefix = prefix
    for i, token_id in enumerate(continuation_tokens):
        try:
            logprobs, cache = model._process_prompt(current_prefix)
            probs = mx.softmax(logprobs[0], axis=-1)
            top_k_indices = mx.argsort(-probs, axis=-1)[:k]
            token_prob = float(
                probs[token_id]) if token_id < probs.shape[0] else 0.0
            k_index = -1
            for rank, idx in enumerate(top_k_indices.tolist()):
                if idx == token_id:
                    k_index = rank
                    break
            token_str = model.tokenizer.decode([token_id]).strip()
            results.append((token_str, k_index, token_prob))
            current_prefix = current_prefix + (token_id,)
        except Exception as e:
            logger.warning(
                f"Error processing token at position {i}: {e}. Skipping.")
            continue
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

# Compute top 10 words and sequences
top_k = 10
top_words = compute_top_confident_words(model, context, top_k=top_k)
print(f"\nTop 10 next words and their confidences:")
for k, word, prob in top_words:
    print(f"k-index: {k}, Word: {word}, Confidence: {prob:.3f}")

top_sequences = compute_top_sequences(model, context)
print(f"\nTop 10 sequences and their average confidences:")
for start_word, seq, conf in top_sequences:
    print(f"Sequence: {seq}, Average Confidence: {conf:.3f}")

# Compute k index and probability for each token in continuation
token_results = compute_token_k_index_and_probability(
    model, context, continuation, k=10)
print(f"\nContinuation tokens' k index and probability:")
for token, k_index, prob in token_results:
    print(f"Token: {token}, k-index: {k_index}, Probability: {prob:.3f}")
