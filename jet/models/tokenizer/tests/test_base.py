import pytest
import numpy as np
from tokenizers import Tokenizer
from jet.models.tokenizer.base import count_tokens, get_tokenizer, get_tokenizer_fn, tokenize


class TestGetTokenizer:
    @pytest.fixture
    def model_name(self):
        return "bert-base-uncased"

    def test_get_tokenizer(self, model_name):
        """Test that get_tokenizer returns a valid Tokenizer instance."""
        result = get_tokenizer(model_name)
        expected = Tokenizer
        assert isinstance(
            result, expected), f"Expected {expected}, but got {type(result)}"


class TestGetTokenizerFn:
    @pytest.fixture
    def model_name(self):
        return "bert-base-uncased"

    def test_get_tokenizer_fn_single_text(self, model_name):
        """Test get_tokenizer_fn with a single text input."""
        tokenizer_fn = get_tokenizer_fn(model_name)
        input_text = "Hello world"
        result = tokenizer_fn(input_text)
        expected = get_tokenizer(model_name).encode(
            input_text, add_special_tokens=True).ids
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_get_tokenizer_fn_multiple_texts(self, model_name):
        """Test get_tokenizer_fn with multiple text inputs."""
        tokenizer_fn = get_tokenizer_fn(model_name)
        input_texts = ["Hello world", "Test sentence"]
        result = tokenizer_fn(input_texts)
        expected = [
            get_tokenizer(model_name).encode(text, add_special_tokens=True).ids
            for text in input_texts
        ]
        assert result == expected, f"Expected {expected}, but got {result}"


class TestTokenize:
    @pytest.fixture
    def model_name(self):
        return "bert-base-uncased"

    def test_tokenize_single_text(self, model_name):
        """Test tokenize function with a single text input."""
        input_text = "Hello world"
        result = tokenize(input_text, model_name)
        expected = get_tokenizer(model_name).encode(
            input_text, add_special_tokens=True).ids
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_tokenize_multiple_texts(self, model_name):
        """Test tokenize function with multiple text inputs."""
        input_texts = ["Hello world", "Test sentence"]
        result = tokenize(input_texts, model_name)
        expected = [
            get_tokenizer(model_name).encode(text, add_special_tokens=True).ids
            for text in input_texts
        ]
        assert result == expected, f"Expected {expected}, but got {result}"


class TestCountTokens:
    @pytest.fixture
    def tokenizer_name(self):
        return "bert-base-uncased"

    def test_count_tokens_empty_input(self, tokenizer_name):
        """Test count_tokens with empty input."""
        messages = []
        result = count_tokens(tokenizer_name, messages)
        expected = 0
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_count_tokens_single_string(self, tokenizer_name):
        """Test count_tokens with a single string input."""
        messages = "Hello world"
        tokenizer = get_tokenizer(tokenizer_name)
        expected = len(tokenizer.encode(messages, add_special_tokens=True).ids)
        result = count_tokens(tokenizer_name, messages)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_count_tokens_list_of_strings(self, tokenizer_name):
        """Test count_tokens with a list of strings, summing token counts."""
        messages = ["Hello world", "Test sentence"]
        tokenizer = get_tokenizer(tokenizer_name)
        expected = sum(
            len(tokenizer.encode(text, add_special_tokens=True).ids)
            for text in messages
        )
        result = count_tokens(tokenizer_name, messages)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_count_tokens_list_of_strings_prevent_total(self, tokenizer_name):
        """Test count_tokens with a list of strings and prevent_total=True."""
        messages = ["Hello world", "Test sentence"]
        tokenizer = get_tokenizer(tokenizer_name)
        expected = [
            len(tokenizer.encode(text, add_special_tokens=True).ids)
            for text in messages
        ]
        result = count_tokens(tokenizer_name, messages, prevent_total=True)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_count_tokens_list_of_dicts(self, tokenizer_name):
        """Test count_tokens with a list of dictionaries, summing token counts."""
        messages = [{"text": "Hello world"}, {"text": "Test sentence"}]
        tokenizer = get_tokenizer(tokenizer_name)
        expected = sum(
            len(tokenizer.encode(str(text), add_special_tokens=True).ids)
            for text in messages
        )
        result = count_tokens(tokenizer_name, messages)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_count_tokens_list_of_dicts_prevent_total(self, tokenizer_name):
        """Test count_tokens with a list of dictionaries and prevent_total=True."""
        messages = [{"text": "Hello world"}, {"text": "Test sentence"}]
        tokenizer = get_tokenizer(tokenizer_name)
        expected = [
            len(tokenizer.encode(str(text), add_special_tokens=True).ids)
            for text in messages
        ]
        result = count_tokens(tokenizer_name, messages, prevent_total=True)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_count_tokens_with_tokenizer_instance(self, tokenizer_name):
        """Test count_tokens with a Tokenizer instance."""
        messages = "Hello world"
        tokenizer_instance = get_tokenizer(tokenizer_name)
        expected = len(tokenizer_instance.encode(
            messages, add_special_tokens=True).ids)
        result = count_tokens(tokenizer_instance, messages)
        assert result == expected, f"Expected {expected}, but got {result}"
