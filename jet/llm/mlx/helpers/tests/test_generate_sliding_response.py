from jet.llm.mlx.generation import stream_chat
from jet.llm.mlx.helpers.generate_sliding_response import (
    count_tokens,
    trim_context,
    estimate_remaining_tokens,
    sliding_window,
    generate_sliding_response,
)
from jet.models.model_types import LLMModelType
from jet.llm.mlx.models import resolve_model
from jet.llm.mlx.token_utils import get_tokenizer
import pytest
from mlx_lm import load
import json
import uuid


DEFAULT_MODEL: LLMModelType = "qwen3-1.7b-4bit"


class TestGenerateSlidingWindow:
    def setup_method(self):
        # Mock tokenizer for testing
        class MockTokenizer:
            def encode(self, text):
                # Approximate token count as word count
                return [0] * len(text.split())

            def apply_chat_template(self, messages, add_generation_prompt):
                return json.dumps(messages)

        self.tokenizer = MockTokenizer()

    def test_trim_context(self):
        messages = [
            {"role": "system", "content": "System instruction"},
            {"role": "user", "content": "First query"},
            {"role": "assistant", "content": "First response"},
            {"role": "user", "content": "Second query"},
            {"role": "assistant", "content": "Second response"},
            {"role": "user", "content": "Third query"}
        ]
        max_context_tokens = 15  # Small limit to force trimming
        expected_messages = [
            {"role": "system", "content": "System instruction"},
            {"role": "assistant", "content": "Second response"},
            {"role": "user", "content": "Third query"}
        ]
        # Approximate: "System instruction" (2) + "Second response" (2) + "Third query" (2)
        expected_tokens = 15

        result_messages, result_tokens = trim_context(
            messages, max_context_tokens, self.tokenizer, preserve_system=True)

        assert result_messages == expected_messages
        assert result_tokens == expected_tokens

    def test_estimate_remaining_tokens(self):
        messages = [
            {"role": "system", "content": "System instruction"},
            {"role": "user", "content": "User query"}
        ]
        context_window = 100
        expected_remaining = 90  # 100 - (2 for system + 2 for user)

        result_remaining = estimate_remaining_tokens(
            messages, context_window, self.tokenizer)

        assert result_remaining == expected_remaining

    def test_generate_sliding_response(self, monkeypatch):
        # Mock stream_chat to return controlled chunks
        def mock_stream_chat(messages, model, max_tokens, temperature, top_p, verbose):
            yield {"choices": [{"message": {"content": "Generated response chunk"}}]}

        monkeypatch.setattr(
            "jet.llm.mlx.generation.stream_chat", mock_stream_chat)

        # Mock get_tokenizer to return our mock tokenizer
        def mock_get_tokenizer(model):
            return self.tokenizer

        monkeypatch.setattr(
            "jet.llm.mlx.token_utils.get_tokenizer", mock_get_tokenizer)

        # Input data
        messages = [
            {"role": "system", "content": "System instruction"},
            {"role": "user", "content": "Test query"}
        ]
        max_tokens_per_generation = 100
        context_window = 500

        # Expected output
        expected_response = "Sure! Please provide the query you'd like me to test."
        expected_messages_after = [
            {"role": "system", "content": "System instruction"},
            {"role": "user", "content": "Test query"},
            {"role": "assistant", "content": expected_response}
        ]

        # Run the function
        result_response = generate_sliding_response(
            messages, max_tokens_per_generation, context_window, DEFAULT_MODEL
        )

        # Assert results
        assert result_response == expected_response
        assert messages == expected_messages_after  # Messages list is modified in-place


class TestCountTokens:
    def test_count_tokens_empty_string(self):
        tokenizer = get_tokenizer(DEFAULT_MODEL)
        text = ""
        expected = 0
        result = count_tokens(text, tokenizer)
        assert result == expected, f"Expected {expected} tokens, but got {result}"

    def test_count_tokens_simple_text(self):
        tokenizer = get_tokenizer(DEFAULT_MODEL)
        text = "Hello, world!"
        expected = len(tokenizer.encode(text))
        result = count_tokens(text, tokenizer)
        assert result == expected, f"Expected {expected} tokens, but got {result}"


class TestTrimContext:
    def test_trim_context_no_trimming_needed(self):
        tokenizer = get_tokenizer(DEFAULT_MODEL)
        messages = [{"role": "system", "content": "System prompt"}, {
            "role": "user", "content": "Hello"}]
        max_context_tokens = 1000
        expected_messages = messages
        expected_tokens = sum(count_tokens(
            json.dumps(msg), tokenizer) for msg in messages)
        result_messages, result_tokens = trim_context(
            messages, max_context_tokens, tokenizer)
        assert result_messages == expected_messages, f"Expected messages {expected_messages}, but got {result_messages}"
        assert result_tokens == expected_tokens, f"Expected {expected_tokens} tokens, but got {result_tokens}"

    def test_trim_context_preserve_system(self):
        tokenizer = get_tokenizer(DEFAULT_MODEL)
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "First message"},
            {"role": "user", "content": "Second message"}
        ]
        max_context_tokens = count_tokens(json.dumps(
            messages[0]), tokenizer) + count_tokens(json.dumps(messages[1]), tokenizer)
        expected_messages = [messages[0], messages[1]]
        expected_tokens = max_context_tokens
        result_messages, result_tokens = trim_context(
            messages, max_context_tokens, tokenizer, preserve_system=True)
        assert result_messages == expected_messages, f"Expected messages {expected_messages}, but got {result_messages}"
        assert result_tokens == expected_tokens, f"Expected {expected_tokens} tokens, but got {result_tokens}"


class TestEstimateRemainingTokens:
    def test_estimate_remaining_tokens(self):
        tokenizer = get_tokenizer(DEFAULT_MODEL)
        messages = [{"role": "user", "content": "Hello"}]
        context_window = 1000
        expected = context_window - \
            count_tokens(json.dumps(messages[0]), tokenizer)
        result = estimate_remaining_tokens(messages, context_window, tokenizer)
        assert result == expected, f"Expected {expected} remaining tokens, but got {result}"


class TestSlidingWindow:
    def test_sliding_window_no_cutoff(self):
        tokenizer = get_tokenizer(DEFAULT_MODEL)
        messages = [{"role": "user", "content": "Hello"}]
        max_tokens_per_generation = 100
        context_window = 1000
        response_chunk = "Response"
        cutoff_detected = False
        expected_messages = [{"role": "user", "content": "Hello"}]
        expected_max_tokens = min(max_tokens_per_generation, estimate_remaining_tokens(
            messages, context_window, tokenizer))
        expected_total_tokens = sum(count_tokens(
            json.dumps(msg), tokenizer) for msg in messages)
        result_messages, result_max_tokens, result_total_tokens = sliding_window(
            messages, max_tokens_per_generation, context_window, tokenizer, response_chunk, cutoff_detected
        )
        assert result_messages == expected_messages, f"Expected messages {expected_messages}, but got {result_messages}"
        assert result_max_tokens == expected_max_tokens, f"Expected {expected_max_tokens} max tokens, but got {result_max_tokens}"
        assert result_total_tokens == expected_total_tokens, f"Expected {expected_total_tokens} total tokens, but got {result_total_tokens}"

    def test_sliding_window_with_cutoff(self):
        tokenizer = get_tokenizer(DEFAULT_MODEL)
        messages = [{"role": "user", "content": "Hello"}, {
            "role": "assistant", "content": "Response\n[CONTINUE]"}]
        max_tokens_per_generation = 100
        context_window = 1000
        response_chunk = "Response\n[CONTINUE]"
        cutoff_detected = True
        expected_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "Continue the previous response where it left off."}
        ]
        expected_max_tokens = min(max_tokens_per_generation, estimate_remaining_tokens(
            expected_messages, context_window, tokenizer))
        expected_total_tokens = sum(count_tokens(
            json.dumps(msg), tokenizer) for msg in expected_messages)
        result_messages, result_max_tokens, result_total_tokens = sliding_window(
            messages, max_tokens_per_generation, context_window, tokenizer, response_chunk, cutoff_detected
        )
        assert result_messages == expected_messages, f"Expected messages {expected_messages}, but got {result_messages}"
        assert result_max_tokens == expected_max_tokens, f"Expected {expected_max_tokens} max tokens, but got {result_max_tokens}"
        assert result_total_tokens == expected_total_tokens, f"Expected {expected_total_tokens} total tokens, but got {result_total_tokens}"

    def test_sliding_window_insufficient_tokens(self):
        tokenizer = get_tokenizer(DEFAULT_MODEL)
        messages = [{"role": "user", "content": "A" * 1000}]
        max_tokens_per_generation = 100
        context_window = 500
        response_chunk = "Response"
        cutoff_detected = False
        expected_messages, expected_total_tokens = trim_context(
            messages, context_window - max_tokens_per_generation, tokenizer)
        expected_max_tokens = min(max_tokens_per_generation, estimate_remaining_tokens(
            expected_messages, context_window, tokenizer))
        result_messages, result_max_tokens, result_total_tokens = sliding_window(
            messages, max_tokens_per_generation, context_window, tokenizer, response_chunk, cutoff_detected
        )
        assert result_messages == expected_messages, f"Expected messages {expected_messages}, but got {result_messages}"
        assert result_max_tokens == expected_max_tokens, f"Expected {expected_max_tokens} max tokens, but got {result_max_tokens}"
        assert result_total_tokens == expected_total_tokens, f"Expected {expected_total_tokens} total tokens, but got {result_total_tokens}"
