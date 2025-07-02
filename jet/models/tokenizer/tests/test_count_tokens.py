import pytest
import numpy as np
from typing import Union, List, Dict
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from tokenizers import Tokenizer
from jet.models.tokenizer.base import count_tokens, get_tokenizer_fn


@pytest.fixture
def tokenizer() -> PreTrainedTokenizerBase:
    """Fixture to provide a tokenizer for tests."""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.pad_token = tokenizer.pad_token or "[PAD]"
    tokenizer.pad_token_id = tokenizer.pad_token_id or 0
    return tokenizer


@pytest.fixture
def tokenizer_no_pad() -> PreTrainedTokenizerBase:
    """Fixture to provide a tokenizer with no pad_token_id."""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.pad_token = None
    tokenizer.pad_token_id = None
    return tokenizer


class TestCountTokens:
    def test_count_tokens_single_string(self, tokenizer: PreTrainedTokenizerBase):
        """Test count_tokens with a single string input."""
        # Given
        text = "Hello world"
        expected = len(tokenizer.encode(text, add_special_tokens=True))

        # When
        result = count_tokens(tokenizer, text, remove_pad_tokens=False)

        # Then
        assert result == expected

    def test_count_tokens_list_of_strings(self, tokenizer: PreTrainedTokenizerBase):
        """Test count_tokens with a list of strings."""
        # Given
        texts = ["Hello world", "This is a test"]
        expected = [len(tokenizer.encode(text, add_special_tokens=True))
                    for text in texts]

        # When
        result = count_tokens(
            tokenizer, texts, prevent_total=False, remove_pad_tokens=False)

        # Then
        assert result == sum(expected)

    def test_count_tokens_list_prevent_total(self, tokenizer: PreTrainedTokenizerBase):
        """Test count_tokens with prevent_total=True."""
        # Given
        texts = ["Hello world", "This is a test"]
        expected = [len(tokenizer.encode(text, add_special_tokens=True))
                    for text in texts]

        # When
        result = count_tokens(
            tokenizer, texts, prevent_total=True, remove_pad_tokens=False)

        # Then
        assert result == expected

    def test_count_tokens_empty_input(self, tokenizer: PreTrainedTokenizerBase):
        """Test count_tokens with empty input."""
        # Given
        texts: Union[str, List[str], List[Dict]] = []
        expected = 0

        # When
        result = count_tokens(tokenizer, texts, remove_pad_tokens=False)

        # Then
        assert result == expected

    def test_count_tokens_with_padding_removal(self, tokenizer: PreTrainedTokenizerBase):
        """Test count_tokens with remove_pad_tokens=True."""
        # Given
        text = "Hello world [PAD] [PAD]"
        tokenize_fn = get_tokenizer_fn(tokenizer, remove_pad_tokens=True)
        expected = len(tokenize_fn(text))

        # When
        result = count_tokens(tokenizer, text, remove_pad_tokens=True)

        # Then
        assert result == expected

    def test_count_tokens_list_with_padding_removal(self, tokenizer: PreTrainedTokenizerBase):
        """Test count_tokens with a list of strings and remove_pad_tokens=True."""
        # Given
        texts = ["Hello world [PAD]", "Test [PAD] [PAD]"]
        tokenize_fn = get_tokenizer_fn(tokenizer, remove_pad_tokens=True)
        expected = [len(tokenize_fn(text)) for text in texts]

        # When
        result = count_tokens(
            tokenizer, texts, prevent_total=True, remove_pad_tokens=True)

        # Then
        assert result == expected

    def test_count_tokens_dict_input(self, tokenizer: PreTrainedTokenizerBase):
        """Test count_tokens with a list of dictionaries."""
        # Given
        messages = [
            {"text": "Hello world"},
            {"text": "This is a test"}
        ]
        expected = [len(tokenizer.encode(str(msg["text"]),
                        add_special_tokens=True)) for msg in messages]

        # When
        result = count_tokens(tokenizer, messages,
                              prevent_total=True, remove_pad_tokens=False)

        # Then
        assert result == expected

    def test_count_tokens_dict_with_padding_removal(self, tokenizer: PreTrainedTokenizerBase):
        """Test count_tokens with a list of dictionaries and remove_pad_tokens=True."""
        # Given
        messages = [
            {"text": "Hello world [PAD]"},
            {"text": "Test [PAD] [PAD]"}
        ]
        tokenize_fn = get_tokenizer_fn(tokenizer, remove_pad_tokens=True)
        expected = [len(tokenize_fn(str(msg["text"]))) for msg in messages]

        # When
        result = count_tokens(tokenizer, messages,
                              prevent_total=True, remove_pad_tokens=True)

        # Then
        assert result == expected

    def test_count_tokens_no_pad_token_id(self, tokenizer_no_pad: PreTrainedTokenizerBase):
        """Test count_tokens with a tokenizer that has no pad_token_id."""
        # Given
        text = "Hello world"
        # Simulate a text with token ID 0 (default pad_token_id)
        encoded = tokenizer_no_pad.encode(text, add_special_tokens=True)
        # Default pad_token_id = 0
        expected = len([tid for tid in encoded if tid != 0])

        # When
        result = count_tokens(tokenizer_no_pad, text, remove_pad_tokens=True)

        # Then
        assert result == expected
