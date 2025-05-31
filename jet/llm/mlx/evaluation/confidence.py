from typing import List, Tuple
from jet.llm.mlx.mlx_types import LLMModelType
from jet.llm.mlx.models import resolve_model
import mlx.core as mx
from mlx_lm.evaluate import MLXLM
from lm_eval.api.instance import Instance


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


# Load model
model_name: LLMModelType = "qwen3-1.7b-4bit"
model = MLXLM(resolve_model(model_name), max_tokens=2048)


# Test data for confidence
context = "The quick brown fox"
continuation = " jumps over the lazy dog."
confidences, is_greedy = compute_confidence(model, context, continuation)
print(f"\nContext: {context}")
print(f"Continuation: {continuation}")
print(f"Token confidences: {[f'{c:.3f}' for c in confidences]}")
print(f"Is greedy: {is_greedy}")
