import pytest
import numpy as np
from typing import List, Union
from jet.models.embeddings.base import get_embedding_function
from io import StringIO
import sys


class TestEmbeddingTokenizer:
    def test_bert_tokenizer_single(self):
        model_name = "bert-base-cased"
        input_text = "I can feel the magic, can you?"
        expected_tokens = [101, 146, 1169, 1631,
                           1103, 3974, 117, 1169, 1128, 136, 102]

        tokenize_fn = get_embedding_function(model_name)
        result = tokenize_fn(input_text)

        assert result == expected_tokens, f"Expected {expected_tokens}, got {result}"

    def test_sentence_transformer_batch(self):
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        input_texts = ["This is a test.", "Another sentence."]
        expected_tokens = [
            [101, 2023, 2003, 1037, 3231, 1012, 102],
            [101, 2178, 6251, 1012, 102]
        ]

        tokenize_fn = get_embedding_function(model_name)
        result = tokenize_fn(input_texts)

        assert result == expected_tokens, f"Expected {expected_tokens}, got {result}"

    def test_gpt2_tokenizer_single(self):
        model_name = "gpt2"
        input_text = "The quick brown fox jumps."
        expected_tokens = [464, 2066, 7586, 4419, 13920, 13]

        tokenize_fn = get_embedding_function(model_name)
        result = tokenize_fn(input_text)

        assert result == expected_tokens, f"Expected {expected_tokens}, got {result}"

    def test_progress_tracking(self, capsys):
        model_name = "bert-base-cased"
        input_texts = ["Test sentence."] * 100
        tokenize_fn = get_embedding_function(model_name, show_progress=True)

        with capsys.disabled():  # Suppress tqdm output for clean test
            result = tokenize_fn(input_texts)

        assert len(
            result) == 100, f"Expected 100 tokenized outputs, got {len(result)}"
        assert result[0] == [101, 1332, 6251, 1012,
                             102], f"Expected [101, 1332, 6251, 1012, 102], got {result[0]}"

    @pytest.mark.benchmark(group="tokenization")
    def test_bert_performance_single(self, benchmark):
        model_name = "bert-base-cased"
        input_text = "I can feel the magic, can you?" * 10  # Longer text
        tokenize_fn = get_embedding_function(model_name)
        result = benchmark(tokenize_fn, input_text)
        expected_tokens = [101, 146, 1169, 1631, 1103,
                           3974, 117, 1169, 1128, 136] * 10 + [102]
        assert result == expected_tokens, f"Expected {expected_tokens}, got {result}"

    @pytest.mark.benchmark(group="tokenization")
    def test_sentence_transformer_performance_batch(self, benchmark):
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        input_texts = ["This is a test sentence for embedding."] * 1000
        tokenize_fn = get_embedding_function(model_name, batch_size=100)
        result = benchmark(tokenize_fn, input_texts)
        assert len(
            result) == 1000, f"Expected 1000 tokenized outputs, got {len(result)}"
        assert result[0] == [101, 2023, 2003, 1037,
                             3231, 6251, 2005, 7861, 4667, 1012, 102]

    @pytest.mark.benchmark(group="tokenization")
    def test_gpt2_performance_dynamic_batch(self, benchmark):
        model_name = "gpt2"
        input_texts = ["The quick brown fox jumps over the lazy dog."] * 1000
        tokenize_fn = get_embedding_function(model_name)  # Dynamic batch size
        result = benchmark(tokenize_fn, input_texts)
        assert len(
            result) == 1000, f"Expected 1000 tokenized outputs, got {len(result)}"
        assert result[0] == [464, 2066, 7586, 4419,
                             13920, 625, 1103, 13971, 3290, 13]
