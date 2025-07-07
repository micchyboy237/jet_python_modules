import unittest
import numpy as np
from clustering_examples import (
    group_similar_texts_agglomerative,
    group_similar_texts_kmeans,
    group_similar_texts_dbscan,
    group_similar_texts_hdbscan,
    group_similar_texts_spectral,
    group_similar_texts_gmm,
)


class TestClusteringExamples(unittest.TestCase):
    def setUp(self):
        # Sample texts and embeddings for testing
        self.texts = [
            "I love coding",
            "Coding is fun",
            "I enjoy programming",
            "The sky is blue"
        ]
        self.embeddings = [np.random.rand(768) for _ in range(len(self.texts))]
        # Normalize embeddings for consistency
        embeddings_array = np.array(self.embeddings)
        norm = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        self.embeddings = (embeddings_array / np.maximum(norm, 1e-10)).tolist()

    def test_agglomerative_empty_input(self):
        result = group_similar_texts_agglomerative([])
        self.assertEqual(result, [], "Empty input should return empty list")

    def test_agglomerative_single_text(self):
        result = group_similar_texts_agglomerative(
            ["I love coding"], embeddings=[self.embeddings[0]])
        self.assertEqual(result, [["I love coding"]],
                         "Single text should return single cluster")

    def test_agglomerative_deduplication(self):
        texts = ["I love coding", "I love coding", "Coding is fun"]
        result = group_similar_texts_agglomerative(
            texts, embeddings=self.embeddings[:3])
        self.assertEqual(
            len(result), 2, "Deduplicated texts should form two clusters")
        self.assertTrue(all(len(cluster) <= 2 for cluster in result),
                        "Deduplication should reduce duplicates")

    def test_agglomerative_with_embeddings(self):
        result = group_similar_texts_agglomerative(
            self.texts, threshold=0.7, embeddings=self.embeddings)
        self.assertIsInstance(result, list, "Result should be a list")
        self.assertTrue(all(isinstance(cluster, list)
                        for cluster in result), "Each cluster should be a list")
        self.assertTrue(all(isinstance(text, str)
                        for cluster in result for text in cluster), "Clusters should contain strings")
        self.assertEqual(
            len(set(tuple(cluster) for cluster in result)),
            len(result),
            "Clusters should be unique"
        )

    def test_kmeans_empty_input(self):
        result = group_similar_texts_kmeans([])
        self.assertEqual(result, [], "Empty input should return empty list")

    def test_kmeans_single_text(self):
        result = group_similar_texts_kmeans(
            ["I love coding"], n_clusters=1, embeddings=[self.embeddings[0]])
        self.assertEqual(result, [["I love coding"]],
                         "Single text should return single cluster")

    def test_kmeans_deduplication(self):
        texts = ["I love coding", "I love coding", "Coding is fun"]
        result = group_similar_texts_kmeans(
            texts, n_clusters=2, embeddings=self.embeddings[:3])
        self.assertEqual(
            len(result), 2, "Deduplicated texts should form two clusters")

    def test_kmeans_with_embeddings(self):
        result = group_similar_texts_kmeans(
            self.texts, n_clusters=2, embeddings=self.embeddings)
        self.assertIsInstance(result, list, "Result should be a list")
        self.assertTrue(all(isinstance(cluster, list)
                        for cluster in result), "Each cluster should be a list")
        self.assertTrue(all(isinstance(text, str)
                        for cluster in result for text in cluster), "Clusters should contain strings")
        self.assertEqual(
            len(result), 2, "Should form exactly n_clusters clusters")

    def test_dbscan_empty_input(self):
        result = group_similar_texts_dbscan([])
        self.assertEqual(result, [], "Empty input should return empty list")

    def test_dbscan_single_text(self):
        result = group_similar_texts_dbscan(
            ["I love coding"], eps=0.3, min_samples=2, embeddings=[self.embeddings[0]])
        self.assertEqual(
            result, [], "Single text with min_samples=2 should return no clusters")

    def test_dbscan_with_embeddings(self):
        result = group_similar_texts_dbscan(
            self.texts, eps=0.3, min_samples=2, embeddings=self.embeddings)
        self.assertIsInstance(result, list, "Result should be a list")
        self.assertTrue(all(isinstance(cluster, list)
                        for cluster in result), "Each cluster should be a list")
        self.assertTrue(all(isinstance(text, str)
                        for cluster in result for text in cluster), "Clusters should contain strings")

    def test_hdbscan_empty_input(self):
        try:
            result = group_similar_texts_hdbscan([])
            self.assertEqual(
                result, [], "Empty input should return empty list")
        except ImportError:
            self.skipTest("HDBSCAN not installed")

    def test_hdbscan_single_text(self):
        try:
            result = group_similar_texts_hdbscan(
                ["I love coding"], min_cluster_size=2, embeddings=[self.embeddings[0]])
            self.assertEqual(
                result, [], "Single text with min_cluster_size=2 should return no clusters")
        except ImportError:
            self.skipTest("HDBSCAN not installed")

    def test_hdbscan_with_embeddings(self):
        try:
            result = group_similar_texts_hdbscan(
                self.texts, min_cluster_size=2, embeddings=self.embeddings)
            self.assertIsInstance(result, list, "Result should be a list")
            self.assertTrue(all(isinstance(cluster, list)
                            for cluster in result), "Each cluster should be a list")
            self.assertTrue(all(isinstance(
                text, str) for cluster in result for text in cluster), "Clusters should contain strings")
        except ImportError:
            self.skipTest("HDBSCAN not installed")

    def test_spectral_empty_input(self):
        result = group_similar_texts_spectral([])
        self.assertEqual(result, [], "Empty input should return empty list")

    def test_spectral_single_text(self):
        result = group_similar_texts_spectral(
            ["I love coding"], n_clusters=1, embeddings=[self.embeddings[0]])
        self.assertEqual(result, [["I love coding"]],
                         "Single text should return single cluster")

    def test_spectral_with_embeddings(self):
        result = group_similar_texts_spectral(
            self.texts, n_clusters=2, embeddings=self.embeddings)
        self.assertIsInstance(result, list, "Result should be a list")
        self.assertTrue(all(isinstance(cluster, list)
                        for cluster in result), "Each cluster should be a list")
        self.assertTrue(all(isinstance(text, str)
                        for cluster in result for text in cluster), "Clusters should contain strings")
        self.assertEqual(
            len(result), 2, "Should form exactly n_clusters clusters")

    def test_gmm_empty_input(self):
        result = group_similar_texts_gmm([])
        self.assertEqual(result, [], "Empty input should return empty list")

    def test_gmm_single_text(self):
        result = group_similar_texts_gmm(
            ["I love coding"], n_components=1, embeddings=[self.embeddings[0]])
        self.assertEqual(result, [["I love coding"]],
                         "Single text should return single cluster")

    def test_gmm_with_embeddings(self):
        result = group_similar_texts_gmm(
            self.texts, n_components=2, embeddings=self.embeddings)
        self.assertIsInstance(result, list, "Result should be a list")
        self.assertTrue(all(isinstance(cluster, list)
                        for cluster in result), "Each cluster should be a list")
        self.assertTrue(all(isinstance(text, str)
                        for cluster in result for text in cluster), "Clusters should contain strings")
        self.assertEqual(
            len(result), 2, "Should form exactly n_components clusters")

    def test_invalid_embeddings(self):
        with self.assertRaises(ValueError):
            group_similar_texts_agglomerative(
                self.texts, embeddings=[np.random.rand(3, 4)])  # Invalid shape


if __name__ == "__main__":
    unittest.main()
