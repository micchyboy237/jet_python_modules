import unittest
from jet.llm.mlx.mlx_types import EmbedModelType
import numpy as np
from jet.llm.utils.transformer_embeddings import generate_embeddings, get_embedding_function, search_docs


class TestEmbeddingGeneration(unittest.TestCase):
    def setUp(self):
        self.model_key: EmbedModelType = "all-minilm:22m"
        self.sample_text = "Hello world!"
        self.sample_texts = ["The sky is blue.", "The grass is green."]

    def test_single_string_embedding(self):
        result = generate_embeddings(self.model_key, self.sample_text)
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], float)
        self.assertGreater(len(result), 0)

    def test_multiple_strings_embedding(self):
        expected_length = len(self.sample_texts)
        result = generate_embeddings(self.model_key, self.sample_texts)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), expected_length)
        for vec in result:
            self.assertIsInstance(vec, list)
            self.assertIsInstance(vec[0], float)

    def test_normalization(self):
        result = generate_embeddings(
            self.model_key, self.sample_text, normalize=True)
        norm = sum([v**2 for v in result]) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=2)

    def test_empty_input(self):
        with self.assertRaises(ValueError):
            generate_embeddings(self.model_key, "")
        with self.assertRaises(ValueError):
            generate_embeddings(self.model_key, [])

    def test_invalid_batch_size(self):
        with self.assertRaises(ValueError):
            generate_embeddings(self.model_key, self.sample_text, batch_size=0)

    def test_embedding_function(self):
        embed_func = get_embedding_function(self.model_key)
        result = embed_func(self.sample_text)
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], float)
        self.assertGreater(len(result), 0)

    def test_search_docs(self):
        documents = [
            "The sky is blue today.",
            "The grass is green.",
            "The sun is shining brightly.",
            "Clouds are white and fluffy."
        ]
        query = "blue sky"

        results = search_docs(query, documents, self.model_key, top_k=2)

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], tuple)
        self.assertEqual(len(results[0]), 2)
        self.assertIsInstance(results[0][0], str)
        self.assertIsInstance(results[0][1], float)
        self.assertGreaterEqual(
            results[0][1], results[1][1])  # Check sorted order

        # Verify most relevant document contains query terms
        self.assertIn("blue", results[0][0].lower())

    def test_search_docs_empty_input(self):
        self.assertEqual(search_docs("", [], self.model_key), [])
        self.assertEqual(search_docs("test", [], self.model_key), [])
        self.assertEqual(search_docs("", ["test"], self.model_key), [])

    def test_search_docs_top_k(self):
        documents = ["doc1", "doc2", "doc3"]
        results = search_docs("test", documents, self.model_key, top_k=2)
        self.assertEqual(len(results), 2)

        results = search_docs("test", documents, self.model_key, top_k=5)
        self.assertEqual(len(results), 3)  # Limited by number of documents


if __name__ == "__main__":
    unittest.main()
