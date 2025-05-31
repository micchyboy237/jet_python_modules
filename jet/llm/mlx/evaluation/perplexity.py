from typing import List
from jet.llm.mlx.mlx_types import LLMModelType
from jet.llm.mlx.models import resolve_model
import mlx.core as mx
from mlx_lm.evaluate import MLXLM
from lm_eval.api.instance import Instance


def compute_perplexity(model: MLXLM, texts: List[str]):
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


# Load model
model_name: LLMModelType = "qwen3-1.7b-4bit"
model = MLXLM(resolve_model(model_name), max_tokens=2048)

# Test data
test_data = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is transforming technology."
]

# Compute perplexity
perplexities = compute_perplexity(model, test_data)
for text, perp in zip(test_data, perplexities):
    print(f"Text: {text}\nPerplexity: {perp:.2f}")
