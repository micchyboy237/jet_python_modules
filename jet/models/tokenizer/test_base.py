import pytest
import numpy as np
from tokenizers import Tokenizer
from jet.models.tokenizer.base import get_tokenizer, get_tokenizer_fn, tokenize


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
