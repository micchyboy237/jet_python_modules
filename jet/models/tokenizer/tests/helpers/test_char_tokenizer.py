import pytest
from typing import List, Union

from jet.models.tokenizer.helpers.char_tokenizer import CharTokenizer


class TestCharTokenizer:
    @pytest.fixture
    def tokenizer(self):
        return CharTokenizer()

    def test_tokenize_basic_sentence(self, tokenizer):
        # Given
        input_text = "Hi!"
        expected_tokens = ["H", "i", "!"]

        # When
        result_tokens = tokenizer.tokenize(input_text)

        # Then
        assert result_tokens == expected_tokens, f"Expected {expected_tokens}, but got {result_tokens}"

    def test_tokenize_empty_string(self, tokenizer):
        # Given
        input_text = ""
        expected_tokens: List[str] = []

        # When
        result_tokens = tokenizer.tokenize(input_text)

        # Then
        assert result_tokens == expected_tokens, f"Expected {expected_tokens}, but got {result_tokens}"

    def test_encode_single_text(self, tokenizer):
        # Given
        input_text = "Hi"
        expected_ids = [2, 72, 105, 3]  # <s>, H, i, </s>

        # When
        result_ids = tokenizer.encode(input_text, add_special_tokens=True)

        # Then
        assert result_ids == expected_ids, f"Expected {expected_ids}, but got {result_ids}"

    def test_encode_batch(self, tokenizer):
        # Given
        input_texts = ["Hi", "Ok"]
        expected_ids = [
            [2, 72, 105, 3],
            [2, 79, 107, 3]
        ]

        # When
        result_ids = tokenizer.encode_batch(
            input_texts, add_special_tokens=True)

        # Then
        assert result_ids == expected_ids, f"Expected {expected_ids}, but got {result_ids}"

    def test_decode_single_ids(self, tokenizer):
        # Given
        token_ids = [2, 72, 105, 3]
        expected_string = "Hi"

        # When
        result_string = tokenizer.decode(token_ids, skip_special_tokens=True)

        # Then
        assert result_string == expected_string, f"Expected '{expected_string}', but got '{result_string}'"

    def test_decode_batch_ids(self, tokenizer):
        # Given
        token_ids = [[2, 72, 105, 3], [2, 79, 107, 3]]
        expected_strings = ["Hi", "Ok"]

        # When
        result_strings = tokenizer.decode(token_ids, skip_special_tokens=True)

        # Then
        assert result_strings == expected_strings, f"Expected {expected_strings}, but got {result_strings}"

    def test_convert_ids_to_tokens(self, tokenizer):
        # Given
        token_ids = [2, 72, 105, 3]
        expected_tokens = ["H", "i"]

        # When
        result_tokens = tokenizer.convert_ids_to_tokens(
            token_ids, skip_special_tokens=True)

        # Then
        assert result_tokens == expected_tokens, f"Expected {expected_tokens}, but got {result_tokens}"

    def test_convert_tokens_to_string(self, tokenizer):
        # Given
        tokens = ["H", "i", "!"]
        expected_string = "Hi!"

        # When
        result_string = tokenizer.convert_tokens_to_string(tokens)

        # Then
        assert result_string == expected_string, f"Expected '{expected_string}', but got '{result_string}'"

    def test_add_tokens(self, tokenizer):
        # Given
        new_tokens = ["α", "β"]
        expected_vocab_size = tokenizer.vocab_size + 2
        expected_new_token_id = tokenizer.vocab_size

        # When
        added_count = tokenizer._add_tokens(new_tokens)
        result_vocab_size = tokenizer.vocab_size
        result_new_token_id = tokenizer._vocab.get("α")

        # Then
        assert added_count == 2, f"Expected 2 tokens added, but got {added_count}"
        assert result_vocab_size == expected_vocab_size, f"Expected vocab size {expected_vocab_size}, but got {result_vocab_size}"
        assert result_new_token_id == expected_new_token_id, f"Expected new token ID {expected_new_token_id}, but got {result_new_token_id}"

    def test_special_tokens(self, tokenizer):
        # Given
        expected_pad_token = "<pad>"

        # When
        result_pad_token = tokenizer.pad_token

        # Then
        assert result_pad_token == expected_pad_token, f"Expected pad token {expected_pad_token}, but got {result_pad_token}"
