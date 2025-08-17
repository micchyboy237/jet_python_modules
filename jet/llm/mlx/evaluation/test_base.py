import pytest
from typing import List, Tuple
import mlx.core as mx
from mlx_lm.evaluate import MLXLM
from lm_eval.api.instance import Instance
from jet.models.model_types import LLMModelType
from jet.llm.mlx.models import resolve_model
from jet.llm.mlx.evaluation.base import (
    compute_perplexity,
    compute_confidence,
    compute_top_confident_words_and_sequences,
    compute_token_k_index_and_probability,
)


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


def compute_top_confident_words_and_sequences(model: MLXLM, context: str, max_tokens: int = 10, num_sequences: int = 5) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    prefix = model._tokenize([context])[0]
    logprobs, _ = model._process_prompt(prefix)
    probs = mx.softmax(logprobs[0], axis=-1)
    top_indices = mx.argsort(-probs, axis=-1)[:5]
    top_probs = probs[top_indices].tolist()
    tokenizer = model.tokenizer
    top_words = []
    for idx, prob in zip(top_indices.tolist(), top_probs):
        token = tokenizer.decode([idx])
        if token.strip() and not token.startswith("##"):
            top_words.append((token.strip(), prob))

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
    prefix = model._tokenize([context])[0]
    full_sequence = model._tokenize([context + continuation])[0]
    continuation_tokens = full_sequence[len(prefix):]
    results = []
    logprobs, cache = model._process_prompt(prefix)
    probs = mx.softmax(logprobs[0], axis=-1)

    if continuation_tokens:
        token_id = continuation_tokens[0]
        token_prob = float(probs[token_id])
        top_indices = mx.argsort(-probs, axis=-1)[:k]
        k_index = int(mx.where(top_indices == token_id)[
                      0][0]) if token_id in top_indices else k
        token = model.tokenizer.decode([token_id]).strip()
        results.append((token, k_index, token_prob))

        if len(continuation_tokens) > 1:
            inputs = mx.array(continuation_tokens)[None, :]
            scores, _, _ = model._score_fn(inputs, cache=cache)

            for i, token_id in enumerate(continuation_tokens[1:], 1):
                logprobs = mx.softmax(scores[0, i-1], axis=-1)
                token_prob = float(logprobs[token_id])
                top_indices = mx.argsort(-logprobs, axis=-1)[:k]
                k_index = int(mx.where(top_indices == token_id)[
                              0][0]) if token_id in top_indices else k
                token = model.tokenizer.decode([token_id]).strip()
                results.append((token, k_index, token_prob))

    return results


class TestPerplexity:
    def setup_method(self):
        model_name: LLMModelType = "qwen2-1.5b-instruct-4bit"
        self.model = MLXLM(resolve_model(model_name), max_tokens=2048)
        self.test_data = ["The quick brown fox jumps over the lazy dog."]

    def test_compute_perplexity(self):
        expected_perplexity = [6.73]  # From your previous output
        result_perplexity = compute_perplexity(self.model, self.test_data)
        assert result_perplexity == pytest.approx(expected_perplexity, abs=1.0), \
            f"Expected perplexity {expected_perplexity}, but got {result_perplexity}"


class TestConfidence:
    def setup_method(self):
        model_name: LLMModelType = "qwen2-1.5b-instruct-4bit"
        self.model = MLXLM(resolve_model(model_name), max_tokens=2048)
        self.context = "The quick brown fox"
        self.continuation = " jumps over the lazy dog."

    def test_compute_confidence(self):
        expected_confidences = [0.665, 0.343, 0.080,
                                0.831, 0.132]  # From your previous output
        expected_is_greedy = True  # From your previous output
        result_confidences, result_is_greedy = compute_confidence(
            self.model, self.context, self.continuation)
        assert result_confidences == pytest.approx(expected_confidences, abs=0.1), \
            f"Expected confidences {expected_confidences}, but got {result_confidences}"
        assert result_is_greedy == expected_is_greedy, \
            f"Expected is_greedy {expected_is_greedy}, but got {result_is_greedy}"


class TestTopConfidentWordsAndSequences:
    def setup_method(self):
        model_name: LLMModelType = "qwen2-1.5b-instruct-4bit"
        self.model = MLXLM(resolve_model(model_name), max_tokens=2048)
        self.context = "The quick brown fox"

    def test_compute_top_confident_words_and_sequences(self):
        expected_words = [("jumps", 0.665), ("leaps", 0.1), ("runs", 0.05),
                          ("hops", 0.03), ("sprints", 0.02)]  # Hypothetical
        expected_sequences = [("jumps over the lazy dog.", 0.35),
                              # Hypothetical, partial
                              ("leaps over the fence.", 0.30)]
        result_words, result_sequences = compute_top_confident_words_and_sequences(
            self.model, self.context, max_tokens=5, num_sequences=2)

        result_word_pairs = [(word, prob) for word, prob in result_words]
        result_seq_pairs = [(seq, conf) for seq, conf in result_sequences]

        assert len(result_words) == len(expected_words), \
            f"Expected {len(expected_words)} words, got {len(result_words)}"
        for (exp_word, exp_prob), (res_word, res_prob) in zip(expected_words, result_word_pairs):
            assert res_word == exp_word, f"Expected word {exp_word}, got {res_word}"
            assert res_prob == pytest.approx(exp_prob, abs=0.1), \
                f"Expected prob {exp_prob} for {exp_word}, got {res_prob}"

        assert len(result_sequences) <= len(expected_sequences), \
            f"Expected up to {len(expected_sequences)} sequences, got {len(result_sequences)}"
        for (exp_seq, exp_conf), (res_seq, res_conf) in zip(expected_sequences, result_seq_pairs):
            assert res_seq == exp_seq, f"Expected sequence {exp_seq}, got {res_seq}"
            assert res_conf == pytest.approx(exp_conf, abs=0.1), \
                f"Expected confidence {exp_conf} for {exp_seq}, got {res_conf}"


class TestTokenKIndexAndProbability:
    def setup_method(self):
        model_name: LLMModelType = "qwen2-1.5b-instruct-4bit"
        self.model = MLXLM(resolve_model(model_name), max_tokens=2048)
        self.context = "The quick brown fox"
        self.continuation = " jumps over the lazy dog."

    def test_compute_token_k_index_and_probability(self):
        expected_results = [
            ("jumps", 0, 0.665),
            ("over", 2, 0.343),
            ("the", 3, 0.080),
            ("lazy", 0, 0.831),
            ("dog.", 1, 0.132)
        ]  # Hypothetical, based on previous confidences
        result_results = compute_token_k_index_and_probability(
            self.model, self.context, self.continuation, k=5)

        assert len(result_results) == len(expected_results), \
            f"Expected {len(expected_results)} tokens, got {len(result_results)}"
        for (exp_token, exp_k_index, exp_prob), (res_token, res_k_index, res_prob) in zip(expected_results, result_results):
            assert res_token == exp_token, f"Expected token {exp_token}, got {res_token}"
            assert res_k_index == exp_k_index, f"Expected k-index {exp_k_index} for {exp_token}, got {res_k_index}"
            assert res_prob == pytest.approx(exp_prob, abs=0.1), \
                f"Expected probability {exp_prob} for {exp_token}, got {res_prob}"
