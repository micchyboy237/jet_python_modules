from typing import List
from nltk.tokenize import sent_tokenize, word_tokenize
from jet.models.model_types import ModelType
from jet.wordnet.text_chunker import chunk_sentences, split_sentences
from jet.models.tokenizer.base import get_tokenizer_fn

MODEL_NAME: ModelType = "qwen3-1.7b-4bit"


class TestChunkSentences:
    def test_no_overlap(self):
        # Given: Text with sentences separated by spaces
        input_text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
        # When: Chunking with size 2 and no overlap
        expected = [
            "This is sentence one. This is sentence two.",
            "This is sentence three. This is sentence four."
        ]
        result = chunk_sentences(input_text, chunk_size=2, sentence_overlap=0)
        # Then: Chunks are correctly formed with spaces
        assert result == expected

    def test_with_overlap(self):
        # Given: Text with sentences separated by spaces
        input_text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
        # When: Chunking with size 2 and overlap 1
        expected = [
            "This is sentence one. This is sentence two.",
            "This is sentence two. This is sentence three.",
            "This is sentence three. This is sentence four."
        ]
        result = chunk_sentences(input_text, chunk_size=2, sentence_overlap=1)
        # Then: Chunks include overlap and preserve spaces
        assert result == expected

    def test_no_overlap_with_model(self):
        # Given: Text with sentences for token-based chunking
        input_text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
        tokenize_fn = get_tokenizer_fn(MODEL_NAME)
        sentences = split_sentences(input_text)
        sentence_tokens = [len(tokenize_fn(s)) for s in sentences]
        # When: Chunking with token-based size and no overlap
        expected_chunks = [
            "This is sentence one. This is sentence two.",
            "This is sentence three. This is sentence four."
        ]
        result = chunk_sentences(input_text, chunk_size=sum(
            sentence_tokens[:2]), sentence_overlap=0, model=MODEL_NAME)
        # Then: Chunks are correctly formed based on tokens
        assert result == expected_chunks

    def test_with_overlap_with_model(self):
        # Given: Text with sentences for token-based chunking
        input_text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
        tokenize_fn = get_tokenizer_fn(MODEL_NAME)
        sentences = split_sentences(input_text)
        sentence_tokens = [len(tokenize_fn(s)) for s in sentences]
        # When: Chunking with token-based size and overlap 1
        expected_chunks = [
            "This is sentence one. This is sentence two.",
            "This is sentence two. This is sentence three.",
            "This is sentence three. This is sentence four."
        ]
        result = chunk_sentences(input_text, chunk_size=sum(
            sentence_tokens[:2]), sentence_overlap=1, model=MODEL_NAME)
        # Then: Chunks include overlap and are token-based
        assert result == expected_chunks

    def test_preserve_newlines(self):
        # Given: Text with sentences separated by newlines
        input_text = "This is sentence one.\nThis is sentence two.\nThis is sentence three.\nThis is sentence four."
        # When: Chunking with size 2 and overlap 1
        expected = [
            "This is sentence one.\nThis is sentence two.",
            "This is sentence two.\nThis is sentence three.",
            "This is sentence three.\nThis is sentence four."
        ]
        result = chunk_sentences(input_text, chunk_size=2, sentence_overlap=1)
        # Then: Chunks preserve newlines
        assert result == expected

    def test_mixed_separators(self):
        # Given: Text with mixed separators (newlines and spaces)
        input_text = "This is sentence one.\nThis is sentence two. This is sentence three.\nThis is sentence four."
        # When: Chunking with size 2 and overlap 1
        expected = [
            "This is sentence one.\nThis is sentence two.",
            "This is sentence two. This is sentence three.",
            "This is sentence three.\nThis is sentence four."
        ]
        result = chunk_sentences(input_text, chunk_size=2, sentence_overlap=1)
        # Then: Chunks preserve original separators
        assert result == expected

    def test_preserve_newlines_with_model(self):
        # Given: Text with sentences separated by newlines for token-based chunking
        input_text = "This is sentence one.\nThis is sentence two.\nThis is sentence three.\nThis is sentence four."
        tokenize_fn = get_tokenizer_fn(MODEL_NAME)
        sentences = split_sentences(input_text)
        sentence_tokens = [len(tokenize_fn(s)) for s in sentences]
        # When: Chunking with token-based size and overlap 1
        expected = [
            "This is sentence one.\nThis is sentence two.",
            "This is sentence two.\nThis is sentence three.",
            "This is sentence three.\nThis is sentence four."
        ]
        result = chunk_sentences(input_text, chunk_size=sum(
            sentence_tokens[:2]), sentence_overlap=1, model=MODEL_NAME)
        # Then: Chunks preserve newlines and are token-based
        assert result == expected
