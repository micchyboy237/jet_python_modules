import pytest
import numpy as np
from typing import List, Union
from jet.models.embeddings.base import get_embedding_function, calculate_batch_size, generate_embeddings
from tokenizers import Tokenizer
from io import StringIO
import sys
import psutil


class TestCalculateBatchSize:
    def test_calculate_batch_size_single_text(self):
        input_text = "This is a test sentence."
        expected = min(max(1, int((psutil.virtual_memory().available /
                       (1024 * 1024)) * 0.5 / (len(input_text) * 0.001))), 128)
        result = calculate_batch_size(input_text)
        assert result == expected, f"Expected {expected}, got {result}"

    def test_calculate_batch_size_list_texts(self):
        input_texts = ["This is a test.", "Another sentence."]
        avg_length = sum(len(t) for t in input_texts) / len(input_texts)
        expected = min(max(1, int((psutil.virtual_memory().available /
                       (1024 * 1024)) * 0.5 / (avg_length * 0.001))), 128)
        result = calculate_batch_size(input_texts)
        assert result == expected, f"Expected {expected}, got {result}"

    def test_calculate_batch_size_with_fixed_batch(self):
        input_texts = ["This is a test.", "Another sentence."]
        fixed_batch_size = 32
        expected = 32
        result = calculate_batch_size(input_texts, fixed_batch_size)
        assert result == expected, f"Expected {expected}, got {result}"


class TestGenerateEmbeddings:
    def test_generate_embeddings_bert_single(self):
        model_name = "bert-base-cased"
        input_text = "I can feel the magic, can you?"
        expected = [101, 146, 1169, 1631, 1103,
                    3974, 117, 1169, 1128, 136, 102]
        tokenizer = Tokenizer.from_pretrained(model_name)
        result = generate_embeddings(input_text, tokenizer, batch_size=1)
        assert result == expected, f"Expected {expected}, got {result}"

    def test_generate_embeddings_sentence_transformer_batch(self):
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        input_texts = ["This is a test.", "Another sentence."]
        expected = [
            [101.0, 2023.0, 2003.0, 1037.0, 3231.0, 1012.0, 102.0] + [0.0] * 121,
            [101.0, 2178.0, 6251.0, 1012.0, 102.0] + [0.0] * 123
        ]
        tokenizer = Tokenizer.from_pretrained(model_name)
        result = generate_embeddings(input_texts, tokenizer, batch_size=2)
        assert result == expected, f"Expected {expected}, got {result}"

    def test_generate_embeddings_gpt2_single(self):
        model_name = "gpt2"
        input_text = "The quick brown fox jumps."
        expected = [464.0, 2068.0, 7586.0, 21831.0, 18045.0, 13.0]
        tokenizer = Tokenizer.from_pretrained(model_name)
        result = generate_embeddings(input_text, tokenizer, batch_size=1)
        assert result == expected, f"Expected {expected}, got {result}"

    def test_generate_embeddings_progress_tracking(self, capsys):
        model_name = "bert-base-cased"
        input_texts = ["Test sentence."] * 100
        expected_first = [101.0, 5960.0, 5650.0, 119.0, 102.0]
        tokenizer = Tokenizer.from_pretrained(model_name)
        with capsys.disabled():
            result = generate_embeddings(
                input_texts, tokenizer, batch_size=10, show_progress=True)
        assert result[0] == expected_first, f"Expected {expected_first}, got {result[0]}"


class TestGetEmbeddingFunction:
    def test_bert_tokenizer_single(self):
        model_name = "bert-base-cased"
        input_text = "I can feel the magic, can you?"
        expected = [101, 146, 1169, 1631, 1103,
                    3974, 117, 1169, 1128, 136, 102]
        tokenize_fn = get_embedding_function(model_name)
        result = tokenize_fn(input_text)
        assert result == expected, f"Expected {expected}, got {result}"

    def test_sentence_transformer_batch(self):
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        input_texts = ["This is a test.", "Another sentence."]
        expected = [
            [101.0, 2023.0, 2003.0, 1037.0, 3231.0, 1012.0, 102.0] + [0.0] * 121,
            [101.0, 2178.0, 6251.0, 1012.0, 102.0] + [0.0] * 123
        ]
        tokenize_fn = get_embedding_function(model_name)
        result = tokenize_fn(input_texts)
        assert result == expected, f"Expected {expected}, got {result}"

    def test_gpt2_tokenizer_single(self):
        model_name = "gpt2"
        input_text = "The quick brown fox jumps."
        expected = [464.0, 2068.0, 7586.0, 21831.0, 18045.0, 13.0]
        tokenize_fn = get_embedding_function(model_name)
        result = tokenize_fn(input_text)
        assert result == expected, f"Expected {expected}, got {result}"

    @pytest.mark.benchmark(group="tokenization")
    def test_bert_performance_single(self, benchmark):
        model_name = "bert-base-cased"
        input_text = "I can feel the magic, can you?" * 10
        expected = [101.0, 146.0, 1169.0, 1631.0, 1103.0, 3974.0, 117.0, 1169.0, 1128.0, 136.0,
                    146.0, 1169.0, 1631.0, 1103.0, 3974.0, 117.0, 1169.0, 1128.0, 136.0,
                    146.0, 1169.0, 1631.0, 1103.0, 3974.0, 117.0, 1169.0, 1128.0, 136.0,
                    146.0, 1169.0, 1631.0, 1103.0, 3974.0, 117.0, 1169.0, 1128.0, 136.0,
                    146.0, 1169.0, 1631.0, 1103.0, 3974.0, 117.0, 1169.0, 1128.0, 136.0,
                    146.0, 1169.0, 1631.0, 1103.0, 3974.0, 117.0, 1169.0, 1128.0, 136.0,
                    146.0, 1169.0, 1631.0, 1103.0, 3974.0, 117.0, 1169.0, 1128.0, 136.0,
                    146.0, 1169.0, 1631.0, 1103.0, 3974.0, 117.0, 1169.0, 1128.0, 136.0,
                    146.0, 1169.0, 1631.0, 1103.0, 3974.0, 117.0, 1169.0, 1128.0, 136.0,
                    146.0, 1169.0, 1631.0, 1103.0, 3974.0, 117.0, 1169.0, 1128.0, 136.0, 102.0]
        tokenize_fn = get_embedding_function(model_name)
        result = benchmark(tokenize_fn, input_text)
        assert result == expected, f"Expected {expected}, got {result}"

    @pytest.mark.benchmark(group="tokenization")
    def test_sentence_transformer_performance_batch(self, benchmark):
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        input_texts = ["This is a test sentence for embedding."] * 1000
        expected_first = [101.0, 2023.0, 2003.0, 1037.0, 3231.0, 6251.0,
                          2005.0, 7861.0, 8270.0, 4667.0, 1012.0, 102.0] + [0.0] * 116
        tokenize_fn = get_embedding_function(model_name, batch_size=100)
        result = benchmark(tokenize_fn, input_texts)
        assert result[0] == expected_first, f"Expected {expected_first}, got {result[0]}"

    @pytest.mark.benchmark(group="tokenization")
    def test_gpt2_performance_dynamic_batch(self, benchmark):
        model_name = "gpt2"
        input_texts = ["The quick brown fox jumps over the lazy dog."] * 1000
        expected_first = [464.0, 2068.0, 7586.0, 21831.0,
                          18045.0, 625.0, 262.0, 16931.0, 3290.0, 13.0]
        tokenize_fn = get_embedding_function(model_name)
        result = benchmark(tokenize_fn, input_texts)
        assert result[0] == expected_first, f"Expected {expected_first}, got {result[0]}"
