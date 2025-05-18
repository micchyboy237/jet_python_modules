import unittest
from jet.llm.mlx.mlx_types import EmbedModelType
import numpy as np
from jet.llm.utils.transformer_embeddings import (
    generate_embeddings,
    get_embedding_function,
    search_docs,
    calculate_dynamic_batch_size,
    chunk_texts,
)
import psutil
import os
import torch


class BaseEmbeddingTest(unittest.TestCase):
    """Base class for shared test utilities and setup."""

    def setUp(self):
        self.small_model_key: EmbedModelType = "all-minilm:22m"
        self.large_model_key: EmbedModelType = "mixedbread-ai/mxbai-embed-large-v1"
        self.sample_text = "Hello world!"
        self.sample_texts = ["The sky is blue.", "The grass is green."]
        self.long_text = " ".join(["word"] * 1000)  # 1,000-word document
        self.very_large_text = " ".join(
            ["word"] * 50000)  # 50,000-word document

    def _get_memory_usage(self) -> float:
        """Get current process memory usage in GB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024**3

    def _get_mps_memory(self) -> float:
        """Get current MPS memory usage in GB."""
        return torch.mps.current_allocated_memory() / 1024**3 if torch.backends.mps.is_available() else 0.0


class TestDynamicBatchSize(BaseEmbeddingTest):
    """Tests for calculate_dynamic_batch_size function."""

    def test_calculate_dynamic_batch_size(self):
        available_memory = 18.13 * 1024 * 1024 * 1024
        batch_size = calculate_dynamic_batch_size(
            384, "mps", available_memory, self.small_model_key)
        self.assertGreater(batch_size, 0)
        self.assertLessEqual(batch_size, 64)
        batch_size = calculate_dynamic_batch_size(
            1024, "mps", available_memory, self.large_model_key)
        self.assertGreater(batch_size, 0)
        self.assertLessEqual(batch_size, 16)
        batch_size = calculate_dynamic_batch_size(
            384, "cpu", available_memory, self.small_model_key)
        self.assertGreater(batch_size, 0)
        self.assertLessEqual(batch_size, 64)


class TestTextChunking(BaseEmbeddingTest):
    """Tests for chunk_texts function."""

    def test_chunk_texts_single_text(self):
        text = " ".join(["word"] * 200)
        chunks, doc_indices = chunk_texts(text, max_tokens=128)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(doc_indices), 2)
        self.assertEqual(chunks[0], " ".join(["word"] * 128))
        self.assertEqual(chunks[1], " ".join(["word"] * 72))
        # Both chunks belong to document 0
        self.assertEqual(doc_indices, [0, 0])

    def test_chunk_texts_multiple_texts(self):
        texts = [" ".join(["word"] * 200), "short text"]
        chunks, doc_indices = chunk_texts(texts, max_tokens=128)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(len(doc_indices), 3)
        self.assertEqual(chunks[0], " ".join(["word"] * 128))
        self.assertEqual(chunks[1], " ".join(["word"] * 72))
        self.assertEqual(chunks[2], "short text")
        # First two chunks from doc 0, last from doc 1
        self.assertEqual(doc_indices, [0, 0, 1])

    def test_chunk_texts_short_text(self):
        chunks, doc_indices = chunk_texts("short", max_tokens=128)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(doc_indices), 1)
        self.assertEqual(chunks[0], "short")
        self.assertEqual(doc_indices, [0])  # Single chunk from document 0


class TestEmbeddingGeneration(BaseEmbeddingTest):
    """Tests for generate_embeddings function."""

    def test_single_string_embedding(self):
        result = generate_embeddings(self.small_model_key, self.sample_text)
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], float)
        self.assertGreater(len(result), 0)

    def test_multiple_strings_embedding(self):
        expected_length = len(self.sample_texts)
        result = generate_embeddings(self.small_model_key, self.sample_texts)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), expected_length)
        for vec in result:
            self.assertIsInstance(vec, list)
            self.assertIsInstance(vec[0], float)

    def test_normalization(self):
        result = generate_embeddings(
            self.small_model_key, self.sample_text, normalize=True)
        norm = sum([v**2 for v in result]) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=2)

    def test_empty_input(self):
        with self.assertRaises(ValueError):
            generate_embeddings(self.small_model_key, "")
        with self.assertRaises(ValueError):
            generate_embeddings(self.small_model_key, [])

    def test_invalid_batch_size(self):
        with self.assertRaises(ValueError):
            generate_embeddings(self.small_model_key,
                                self.sample_text, batch_size=0)

    def test_long_text_chunking(self):
        result = generate_embeddings(
            self.small_model_key, self.long_text, max_tokens=128)
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], float)
        self.assertGreater(len(result), 0)

    def test_mixed_precision(self):
        result = generate_embeddings(
            self.small_model_key, self.sample_text, batch_size=4)
        self.assertFalse(np.any(np.isnan(result)))
        self.assertFalse(np.any(np.isinf(result)))

    def test_max_tokens(self):
        result = generate_embeddings(
            self.small_model_key, self.long_text, max_tokens=128)
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], float)
        self.assertGreater(len(result), 0)

    def test_very_large_document_small_model(self):
        initial_memory = self._get_memory_usage()
        initial_mps = self._get_mps_memory()
        result = generate_embeddings(
            self.small_model_key, self.very_large_text, max_tokens=128, batch_size=4)
        final_memory = self._get_memory_usage()
        final_mps = self._get_mps_memory()
        memory_increase = final_memory - initial_memory
        mps_increase = final_mps - initial_mps

        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], float)
        self.assertGreater(len(result), 0)
        self.assertLess(memory_increase, 2.0,
                        f"RAM usage too high: {memory_increase} GB")
        self.assertLess(mps_increase, 10.0,
                        f"MPS memory usage too high: {mps_increase} GB")
        torch.mps.empty_cache()

    def test_very_large_document_large_model(self):
        initial_memory = self._get_memory_usage()
        initial_mps = self._get_mps_memory()
        result = generate_embeddings(
            self.large_model_key, self.very_large_text, max_tokens=128, batch_size=2)
        final_memory = self._get_memory_usage()
        final_mps = self._get_mps_memory()
        memory_increase = final_memory - initial_memory
        mps_increase = final_mps - initial_mps

        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], float)
        self.assertEqual(len(result), 1024,
                         "Expected 1024-dimensional embeddings")
        self.assertLess(memory_increase, 2.0,
                        f"RAM usage too high: {memory_increase} GB")
        self.assertLess(mps_increase, 16.0,
                        f"MPS memory usage too high: {mps_increase} GB")
        torch.mps.empty_cache()


class TestEmbeddingFunction(BaseEmbeddingTest):
    """Tests for get_embedding_function."""

    def test_embedding_function(self):
        embed_func = get_embedding_function(self.small_model_key)
        result = embed_func(self.sample_text)
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], float)
        self.assertGreater(len(result), 0)


class TestDocumentSearch(BaseEmbeddingTest):
    """Tests for search_docs function."""

    def test_search_docs(self):
        documents = [
            "The sky is blue today.",
            "The grass is green.",
            "The sun is shining brightly.",
            "Clouds are white and fluffy."
        ]
        query = "blue sky"
        results = search_docs(query, documents, self.small_model_key, top_k=2)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], tuple)
        self.assertEqual(len(results[0]), 2)
        self.assertIsInstance(results[0][0], str)
        self.assertIsInstance(results[0][1], float)
        self.assertGreaterEqual(results[0][1], results[1][1])
        self.assertIn("blue", results[0][0].lower())

    def test_search_docs_empty_input(self):
        self.assertEqual(search_docs("", [], self.small_model_key), [])
        self.assertEqual(search_docs("test", [], self.small_model_key), [])
        self.assertEqual(search_docs("", ["test"], self.small_model_key), [])

    def test_search_docs_top_k(self):
        documents = ["doc1", "doc2", "doc3"]
        results = search_docs("test", documents, self.small_model_key, top_k=2)
        self.assertEqual(len(results), 2)
        results = search_docs("test", documents, self.small_model_key, top_k=5)
        self.assertEqual(len(results), 3)


if __name__ == "__main__":
    unittest.main()
