import pytest
from typing import List, Union

from jet.models.tokenizer.helpers.word_tokenizer import WordTokenizer


class TestWordTokenizer:
    @pytest.fixture
    def tokenizer(self):
        return WordTokenizer()

    def test_tokenize_basic_sentence(self, tokenizer):
        # Given: A basic sentence with punctuation
        input_text = "Hello world!"
        expected_tokens = ["Hello", "world"]

        # When: Tokenizing the input text
        result_tokens = tokenizer.tokenize(input_text)

        # Then: Expect only words, excluding punctuation
        assert result_tokens == expected_tokens, f"Expected {expected_tokens}, but got {result_tokens}"

    def test_tokenize_with_unknown_words(self, tokenizer):
        # Given: A sentence with unknown words and punctuation
        input_text = "Hello nonexistent word!"
        expected_tokens = ["Hello", "nonexistent", "word"]

        # When: Tokenizing the input text
        result_tokens = tokenizer.tokenize(input_text)

        # Then: Expect words with unknown tokens, excluding punctuation
        assert result_tokens == expected_tokens, f"Expected {expected_tokens}, but got {result_tokens}"

    def test_tokenize_empty_string(self, tokenizer):
        # Given: An empty string
        input_text = ""
        expected_tokens: List[str] = []

        # When: Tokenizing the empty string
        result_tokens = tokenizer.tokenize(input_text)

        # Then: Expect an empty token list
        assert result_tokens == expected_tokens, f"Expected {expected_tokens}, but got {result_tokens}"

    def test_encode_single_text(self, tokenizer):
        # Given: A simple text with words
        input_text = "Hello world"
        expected_ids = [2, tokenizer._vocab.get("Hello", 1), tokenizer._vocab.get(
            "world", 1), 3]  # <s>, Hello, world, </s>

        # When: Encoding the text with special tokens
        result_ids = tokenizer.encode(input_text, add_special_tokens=True)

        # Then: Expect token IDs including special tokens
        assert result_ids == expected_ids, f"Expected {expected_ids}, but got {result_ids}"

    def test_encode_batch(self, tokenizer):
        # Given: A batch of texts
        input_texts = ["Hello world", "Test sentence"]
        expected_ids = [
            [2, tokenizer._vocab.get("Hello", 1),
             tokenizer._vocab.get("world", 1), 3],
            [2, tokenizer._vocab.get("Test", 1),
             tokenizer._vocab.get("sentence", 1), 3]
        ]

        # When: Encoding the batch with special tokens
        result_ids = tokenizer.encode_batch(
            input_texts, add_special_tokens=True)

        # Then: Expect list of token ID lists
        assert result_ids == expected_ids, f"Expected {expected_ids}, but got {result_ids}"

    def test_decode_single_ids(self, tokenizer):
        # Given: Token IDs for a single text
        tokenizer._add_tokens(["Hello"])  # Ensure "Hello" is in vocab
        token_ids = [2, tokenizer._vocab["Hello"], 1, 3]
        expected_string = "Hello <unk>"

        # When: Decoding the token IDs, skipping special tokens
        result_string = tokenizer.decode(token_ids, skip_special_tokens=True)

        # Then: Expect decoded string with words only
        assert result_string == expected_string, f"Expected '{expected_string}', but got '{result_string}'"

    def test_decode_batch_ids(self, tokenizer):
        # Given: Batch of token IDs
        tokenizer._add_tokens(["Hello", "Test"])  # Ensure words are in vocab
        token_ids = [
            [2, tokenizer._vocab["Hello"], 1, 3],
            [2, tokenizer._vocab["Test"], 1, 3]
        ]
        expected_strings = ["Hello <unk>", "Test <unk>"]

        # When: Decoding the batch, skipping special tokens
        result_strings = tokenizer.decode(token_ids, skip_special_tokens=True)

        # Then: Expect list of decoded strings
        assert result_strings == expected_strings, f"Expected {expected_strings}, but got {result_strings}"

    def test_convert_ids_to_tokens(self, tokenizer):
        # Given: Token IDs for a single text
        tokenizer._add_tokens(["Hello"])  # Ensure "Hello" is in vocab
        token_ids = [2, tokenizer._vocab["Hello"], 1, 3]
        expected_tokens = ["Hello", "<unk>"]

        # When: Converting IDs to tokens, skipping special tokens
        result_tokens = tokenizer.convert_ids_to_tokens(
            token_ids, skip_special_tokens=True)

        # Then: Expect tokens without special tokens
        assert result_tokens == expected_tokens, f"Expected {expected_tokens}, but got {result_tokens}"

    def test_convert_tokens_to_string(self, tokenizer):
        # Given: A list of tokens
        tokens = ["Hello", "<unk>"]
        expected_string = "Hello <unk>"

        # When: Converting tokens to a string
        result_string = tokenizer.convert_tokens_to_string(tokens)

        # Then: Expect space-joined string
        assert result_string == expected_string, f"Expected '{expected_string}', but got '{result_string}'"

    def test_convert_tokens_to_ids(self, tokenizer):
        # Given: A list of special tokens
        tokens = ["<pad>", "<unk>", "<s>"]
        expected_ids = [0, 1, 2]

        # When: Converting tokens to IDs
        result_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Then: Expect correct token IDs
        assert result_ids == expected_ids, f"Expected {expected_ids}, but got {result_ids}"

    def test_add_tokens(self, tokenizer):
        # Given: New tokens to add
        new_tokens = ["new", "token"]
        expected_vocab_size = tokenizer.vocab_size + 2
        expected_new_token_id = tokenizer.vocab_size

        # When: Adding new tokens
        added_count = tokenizer._add_tokens(new_tokens)
        result_vocab_size = tokenizer.vocab_size
        result_new_token_id = tokenizer._vocab.get("new")

        # Then: Expect correct number of tokens added and updated vocab
        assert added_count == 2, f"Expected 2 tokens added, but got {added_count}"
        assert result_vocab_size == expected_vocab_size, f"Expected vocab size {expected_vocab_size}, but got {result_vocab_size}"
        assert result_new_token_id == expected_new_token_id, f"Expected new token ID {expected_new_token_id}, but got {result_new_token_id}"

    def test_special_tokens(self, tokenizer):
        # Given: Expected special token values
        expected_pad_token = "<pad>"
        expected_unk_token = "<unk>"
        expected_bos_token = "<s>"
        expected_eos_token = "</s>"

        # When: Retrieving special tokens
        result_pad_token = tokenizer.pad_token
        result_unk_token = tokenizer.unk_token
        result_bos_token = tokenizer.bos_token
        result_eos_token = tokenizer.eos_token

        # Then: Expect special tokens to match
        assert result_pad_token == expected_pad_token, f"Expected pad token {expected_pad_token}, but got {result_pad_token}"
        assert result_unk_token == expected_unk_token, f"Expected unk token {expected_unk_token}, but got {result_unk_token}"
        assert result_bos_token == expected_bos_token, f"Expected bos token {expected_bos_token}, but got {result_bos_token}"
        assert result_eos_token == expected_eos_token, f"Expected eos token {expected_eos_token}, but got {result_eos_token}"
