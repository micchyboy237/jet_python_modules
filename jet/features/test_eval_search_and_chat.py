import time
import unittest
from unittest.mock import patch, AsyncMock
from jet.llm.utils.embeddings import get_embedding_function
import numpy as np
from collections import Counter
from urllib.parse import urlparse
import nltk
import math
from jet.features.eval_search_and_chat import (
    evaluate_search_results,
    evaluate_html_processing,
    evaluate_llm_response,
    evaluate_pipeline,
    save_output,
    reconstruct_nodes
)
from llama_index.core.schema import Document, NodeWithScore
from typing import List, Dict

# Mock nltk data
nltk.download('punkt', quiet=True)


class TestEvaluateSearchResults(unittest.TestCase):
    @patch("worker.save_output")
    def test_normal_case(self, mock_save_output):
        search_results = [
            {"url": "https://example.com/page1", "title": "Example Page",
                "snippet": "This is a test page"},
            {"url": "https://test.com/page2", "title": "Test Page",
                "snippet": "Another test page"},
        ]
        query = "test page"
        output_dir = "/tmp/test_output"

        result = evaluate_search_results(search_results, query, output_dir)

        self.assertEqual(result["total_results"], 2)
        self.assertEqual(result["unique_domains"], 2)
        self.assertGreater(result["url_diversity_score"], 0.0)
        self.assertGreater(result["keyword_overlap_score"], 0.0)
        mock_save_output.assert_called_once_with(
            result, "/tmp/test_output/search_results_evaluation.json")

    @patch("worker.save_output")
    def test_empty_results(self, mock_save_output):
        search_results: List[Dict[str, str]] = []
        query = "test page"
        output_dir = "/tmp/test_output"

        result = evaluate_search_results(search_results, query, output_dir)

        self.assertEqual(result["total_results"], 0)
        self.assertEqual(result["unique_domains"], 0)
        self.assertEqual(result["url_diversity_score"], 0.0)
        self.assertEqual(result["keyword_overlap_score"], 0.0)
        mock_save_output.assert_called_once_with(
            result, "/tmp/test_output/search_results_evaluation.json")

    @patch("worker.save_output")
    def test_empty_query(self, mock_save_output):
        search_results = [{"url": "https://example.com/page1",
                           "title": "Example Page", "snippet": "This is a test page"}]
        query = ""
        output_dir = "/tmp/test_output"

        result = evaluate_search_results(search_results, query, output_dir)

        self.assertEqual(result["total_results"], 1)
        self.assertEqual(result["unique_domains"], 1)
        self.assertEqual(result["keyword_overlap_score"], 0.0)
        mock_save_output.assert_called_once_with(
            result, "/tmp/test_output/search_results_evaluation.json")


class TestEvaluateHtmlProcessing(unittest.TestCase):
    @patch("worker.save_output")
    def test_normal_case(self, mock_save_output):
        query_scores = [0.8, 0.6, 0.4]
        reranked_nodes = [
            {"text": "Doc 1", "score": 0.8, "metadata": {"doc_index": 1}},
            {"text": "Doc 2", "score": 0.6, "metadata": {"doc_index": 2}},
            {"text": "Doc 3", "score": 0.4, "metadata": {"doc_index": 3}},
        ]
        grouped_nodes = reranked_nodes[:2]
        output_dir = "/tmp/test_output"

        result = evaluate_html_processing(
            query_scores, reranked_nodes, grouped_nodes, output_dir)

        self.assertGreater(result["score_distribution_entropy"], 0.0)
        self.assertEqual(result["node_relevance_score"], 0.6)
        self.assertGreater(result["grouped_nodes_coherence"], 0.0)
        self.assertGreater(result["ndcg_at_5"], 0.0)
        mock_save_output.assert_called_once_with(
            result, "/tmp/test_output/html_processing_evaluation.json")

    @patch("worker.save_output")
    def test_empty_inputs(self, mock_save_output):
        result = evaluate_html_processing([], [], [], "/tmp/test_output")

        self.assertEqual(result["score_distribution_entropy"], 0.0)
        self.assertEqual(result["node_relevance_score"], 0.0)
        self.assertEqual(result["grouped_nodes_coherence"], 0.0)
        self.assertEqual(result["ndcg_at_5"], 0.0)
        mock_save_output.assert_called_once()

    @patch("worker.save_output")
    def test_single_grouped_node(self, mock_save_output):
        query_scores = [0.8]
        reranked_nodes = [
            {"text": "Doc 1", "score": 0.8, "metadata": {"doc_index": 1}}]
        grouped_nodes = reranked_nodes
        output_dir = "/tmp/test_output"

        result = evaluate_html_processing(
            query_scores, reranked_nodes, grouped_nodes, output_dir)

        self.assertEqual(result["grouped_nodes_coherence"], 0.0)
        self.assertEqual(result["node_relevance_score"], 0.8)
        mock_save_output.assert_called_once()


class TestEvaluateLlmResponse(unittest.IsolatedAsyncioTestCase):
    @patch("jet.llm.utils.embeddings.get_embedding_function", new=AsyncMock())
    def test_normal_case(self, mock_get_embedding):
        mock_get_embedding.side_effect = [
            np.array([1.0, 0.0]),
            np.array([0.9, 0.1]),
            np.array([0.8, 0.2]),
            np.array([0.9, 0.1]),
            np.array([0.8, 0.2]),
        ]
        result = evaluate_llm_response(
            "test query", "This is the context.", "This is a test response. It is coherent.",
            embed_model="mxbai-embed-large", llm_model="gemma3:4b", output_dir="/tmp/test_output"
        )
        self.assertGreater(result["query_response_similarity"], 0.9)
        self.assertGreater(result["context_response_similarity"], 0.8)
        self.assertGreater(result["response_coherence_score"], 0.8)

    @patch("worker.save_output")
    def test_empty_response(self, mock_save_output):
        result = evaluate_llm_response("test query", "This is the context.", "",
                                       embed_model="mxbai-embed-large", llm_model="gemma3:4b", output_dir="/tmp/test_output")
        self.assertEqual(result["query_response_similarity"], 0.0)
        self.assertEqual(result["context_response_similarity"], 0.0)
        self.assertEqual(result["response_coherence_score"], 0.0)
        mock_save_output.assert_called_once()

    @patch("jet.llm.utils.embeddings.get_embedding_function", return_value=[0.1, 0.2, 0.3])
    def test_get_embedding_single_string(self, mock_get_embedding):
        embed_func = get_embedding_function("mxbai-embed-large")
        result = embed_func("test sentence")
        self.assertEqual(result, [0.1, 0.2, 0.3])
        self.assertTrue(all(isinstance(x, float) for x in result))

    @patch("jet.llm.utils.embeddings.get_embedding_function", return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    def test_get_embedding_list_of_strings(self, mock_get_embedding):
        embed_func = get_embedding_function("mxbai-embed-large")
        result = embed_func(["sentence one", "sentence two"])
        self.assertEqual(result, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])


class TestEvaluatePipeline(unittest.TestCase):
    @patch("worker.save_output")
    @patch("psutil.Process")
    @patch("psutil.cpu_percent")
    def test_normal_case(self, mock_cpu_percent, mock_process, mock_save_output):
        mock_process.return_value.memory_info.return_value.rss = 1024 * 1024 * 100
        mock_cpu_percent.return_value = 50.0
        start_time = time.time() - 2.0
        output_dir = "/tmp/test_output"
        error_count = 1

        result = evaluate_pipeline(start_time, output_dir, error_count)

        self.assertGreater(result["latency_seconds"], 1.9)
        self.assertEqual(result["memory_usage_mb"], 100.0)
        self.assertEqual(result["cpu_usage_percent"], 50.0)
        self.assertEqual(result["error_count"], 1)
        self.assertEqual(result["error_rate"], 0.5)
        mock_save_output.assert_called_once()

    @patch("worker.save_output")
    @patch("psutil.Process")
    @patch("psutil.cpu_percent")
    def test_zero_metrics(self, mock_cpu_percent, mock_process, mock_save_output):
        mock_process.return_value.memory_info.return_value.rss = 0
        mock_cpu_percent.return_value = 0.0
        start_time = time.time()
        output_dir = "/tmp/test_output"
        error_count = 0

        result = evaluate_pipeline(start_time, output_dir, error_count)

        self.assertGreaterEqual(result["latency_seconds"], 0.0)
        self.assertEqual(result["memory_usage_mb"], 0.0)
        self.assertEqual(result["cpu_usage_percent"], 0.0)
        self.assertEqual(result["error_count"], 0)
        self.assertEqual(result["error_rate"], 0.0)
        mock_save_output.assert_called_once()


class TestHelperFunctions(unittest.TestCase):
    @patch("jet.file.utils.save_file")
    def test_save_output(self, mock_save_file):
        save_output({"test": "data"}, "/tmp/test.json")
        mock_save_file.assert_called_once()

    def test_reconstruct_nodes(self):
        nodes = [
            {"text": "Doc 1", "score": 0.8, "metadata": {"doc_index": 1}},
            {"text": "Doc 2", "score": 0.6, "metadata": {"doc_index": 2}},
        ]
        result = reconstruct_nodes(nodes)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], NodeWithScore)
        self.assertEqual(result[0].text, "Doc 1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
