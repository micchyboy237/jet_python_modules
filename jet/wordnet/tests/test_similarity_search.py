import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from typing import List, Dict, Any
from jet.wordnet.similarity import (
    sentence_similarity,
    get_text_groups,
    query_similarity_scores,
    fuse_all_results,
    SimilarityResult,
    DEFAULT_SENTENCE_EMBED_MODEL,
)

# Mock the embedding function to return predictable embeddings


def mock_embedding_function(model_name: str):
    def embed(texts: str | List[str]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        # Return simple embeddings based on text length for testing
        return [[len(text) / 10.0] * 3 for text in texts]
    return embed


@pytest.fixture
def mock_get_embedding_function():
    with patch("jet.llm.utils.transformer_embeddings.get_embedding_function", side_effect=mock_embedding_function) as mock:
        yield mock


@pytest.fixture
def mock_generate_key():
    def generate_key(text: str, query: str | None = None) -> str:
        return f"id_{text[:5]}"
    with patch("jet.data.utils.generate_key", side_effect=generate_key) as mock:
        yield mock


class TestSentenceSimilarity:
    def test_single_string(self, mock_get_embedding_function):
        base_sentence = "hello"
        sentences_to_compare = "world"
        expected = [1 - np.cos(len("hello") / 10.0 *
                               np.ones(3), len("world") / 10.0 * np.ones(3))]
        result = sentence_similarity(
            base_sentence, sentences_to_compare, model_name=DEFAULT_SENTENCE_EMBED_MODEL)
        assert len(result) == len(
            expected), f"Expected {len(expected)} results, got {len(result)}"
        assert all(abs(r - e) < 1e-5 for r, e in zip(result, expected)
                   ), f"Expected {expected}, got {result}"

    def test_list_input(self, mock_get_embedding_function):
        base_sentence = "hello"
        sentences_to_compare = ["world", "hi"]
        expected = [
            1 - np.cos(len("hello") / 10.0 * np.ones(3),
                       len("world") / 10.0 * np.ones(3)),
            1 - np.cos(len("hello") / 10.0 * np.ones(3),
                       len("hi") / 10.0 * np.ones(3)),
        ]
        result = sentence_similarity(
            base_sentence, sentences_to_compare, model_name=DEFAULT_SENTENCE_EMBED_MODEL)
        assert len(result) == len(
            expected), f"Expected {len(expected)} results, got {len(result)}"
        assert all(abs(r - e) < 1e-5 for r, e in zip(result, expected)
                   ), f"Expected {expected}, got {result}"


class TestGetTextGroups:
    def test_empty_input(self):
        with pytest.raises(ValueError) as exc_info:
            get_text_groups([], threshold=0.75)
        expected = "'texts' must be non-empty."
        result = str(exc_info.value)
        assert result == expected, f"Expected error message '{expected}', got '{result}'"

    def test_single_group(self, mock_get_embedding_function):
        texts = ["hello", "hello world", "hi"]
        threshold = 0.8
        expected = [["hello", "hello world"], ["hi"]]
        result = get_text_groups(
            texts, threshold=threshold, model_name=DEFAULT_SENTENCE_EMBED_MODEL)
        assert result == expected, f"Expected {expected}, got {result}"

    def test_multiple_groups(self, mock_get_embedding_function):
        texts = ["cat", "dog", "cats", "dogs"]
        threshold = 0.9
        expected = [["cat", "cats"], ["dog", "dogs"]]
        result = get_text_groups(
            texts, threshold=threshold, model_name=DEFAULT_SENTENCE_EMBED_MODEL)
        assert result == expected, f"Expected {expected}, got {result}"


class TestQuerySimilarityScores:
    def test_single_query_text(self, mock_get_embedding_function, mock_generate_key):
        query = "hello"
        texts = ["world"]
        threshold = 0.0
        model = DEFAULT_SENTENCE_EMBED_MODEL
        expected_score = float(
            np.dot(np.ones(3) * len("hello") / 10.0, np.ones(3) * len("world") / 10.0))
        expected = [{
            "id": "id_world",
            "rank": 1,
            "doc_index": 0,
            "score": expected_score,
            "percent_difference": 0.0,
            "text": "world"
        }]
        result = query_similarity_scores(
            query, texts, threshold, model, "average", None, "cosine")
        assert len(result) == len(
            expected), f"Expected {len(expected)} results, got {len(result)}"
        assert result[0]["id"] == expected[0][
            "id"], f"Expected id {expected[0]['id']}, got {result[0]['id']}"
        assert abs(result[0]["score"] - expected[0]["score"]
                   ) < 1e-5, f"Expected score {expected[0]['score']}, got {result[0]['score']}"
        assert result[0]["rank"] == expected[0][
            "rank"], f"Expected rank {expected[0]['rank']}, got {result[0]['rank']}"
        assert result[0]["percent_difference"] == expected[0][
            "percent_difference"], f"Expected percent_difference {expected[0]['percent_difference']}, got {result[0]['percent_difference']}"
        assert result[0]["text"] == expected[0][
            "text"], f"Expected text {expected[0]['text']}, got {result[0]['text']}"

    def test_invalid_fuse_method(self, mock_get_embedding_function):
        query = "hello"
        texts = ["world"]
        with pytest.raises(ValueError) as exc_info:
            query_similarity_scores(query, texts, fuse_method="invalid")
        expected = "Fusion method must be one of {'average', 'max', 'min'}; got invalid."
        result = str(exc_info.value)
        assert result == expected, f"Expected error message '{expected}', got '{result}'"

    def test_invalid_metrics(self, mock_get_embedding_function):
        query = "hello"
        texts = ["world"]
        with pytest.raises(ValueError) as exc_info:
            query_similarity_scores(query, texts, metrics="invalid")
        expected = "Metrics must be one of {'cosine', 'euclidean', 'dot'}; got invalid."
        result = str(exc_info.value)
        assert result == expected, f"Expected error message '{expected}', got '{result}'"

    def test_empty_inputs(self, mock_get_embedding_function):
        with pytest.raises(ValueError) as exc_info:
            query_similarity_scores([], ["text"])
        expected = "Both query and texts must be non-empty."
        result = str(exc_info.value)
        assert result == expected, f"Expected error message '{expected}', got '{result}'"

        with pytest.raises(ValueError) as exc_info:
            query_similarity_scores(["query"], [])
        expected = "Both query and texts must be non-empty."
        result = str(exc_info.value)
        assert result == expected, f"Expected error message '{expected}', got '{result}'"

    def test_ids_mismatch(self, mock_get_embedding_function):
        query = "hello"
        texts = ["world", "hi"]
        ids = ["id1"]
        with pytest.raises(ValueError) as exc_info:
            query_similarity_scores(query, texts, ids=ids)
        expected = "Length of ids (1) must match length of texts (2)."
        result = str(exc_info.value)
        assert result == expected, f"Expected error message '{expected}', got '{result}'"


class TestFuseAllResults:
    def test_average(self):
        results = [
            {"id": "id1", "query": "q1", "text": "text1", "score": 0.8},
            {"id": "id1", "query": "q2", "text": "text1", "score": 0.6},
            {"id": "id2", "query": "q1", "text": "text2", "score": 0.5},
        ]
        expected = [
            {"id": "id1", "rank": 1, "score": 0.7,
                "percent_difference": 0.0, "text": "text1"},
            {"id": "id2", "rank": 2, "score": 0.5,
                "percent_difference": 28.57, "text": "text2"},
        ]
        result = fuse_all_results(results, method="average")
        assert len(result) == len(
            expected), f"Expected {len(expected)} results, got {len(result)}"
        for r, e in zip(result, expected):
            assert r["id"] == e["id"], f"Expected id {e['id']}, got {r['id']}"
            assert abs(r["score"] - e["score"]
                       ) < 1e-5, f"Expected score {e['score']}, got {r['score']}"
            assert r["rank"] == e["rank"], f"Expected rank {e['rank']}, got {r['rank']}"
            assert abs(r["percent_difference"] - e["percent_difference"]
                       ) < 1e-2, f"Expected percent_difference {e['percent_difference']}, got {r['percent_difference']}"
            assert r["text"] == e["text"], f"Expected text {e['text']}, got {r['text']}"

    def test_max(self):
        results = [
            {"id": "id1", "query": "q1", "text": "text1", "score": 0.8},
            {"id": "id1", "query": "q2", "text": "text1", "score": 0.6},
            {"id": "id2", "query": "q1", "text": "text2", "score": 0.5},
        ]
        expected = [
            {"id": "id1", "rank": 1, "score": 0.8,
                "percent_difference": 0.0, "text": "text1"},
            {"id": "id2", "rank": 2, "score": 0.5,
                "percent_difference": 37.5, "text": "text2"},
        ]
        result = fuse_all_results(results, method="max")
        assert len(result) == len(
            expected), f"Expected {len(expected)} results, got {len(result)}"
        for r, e in zip(result, expected):
            assert r["id"] == e["id"], f"Expected id {e['id']}, got {r['id']}"
            assert abs(r["score"] - e["score"]
                       ) < 1e-5, f"Expected score {e['score']}, got {r['score']}"
            assert r["rank"] == e["rank"], f"Expected rank {e['rank']}, got {r['rank']}"
            assert abs(r["percent_difference"] - e["percent_difference"]
                       ) < 1e-2, f"Expected percent_difference {e['percent_difference']}, got {r['percent_difference']}"
            assert r["text"] == e["text"], f"Expected text {e['text']}, got {r['text']}"

    def test_invalid_method(self):
        results = [
            {"id": "id1", "query": "q1", "text": "text1", "score": 0.8},
        ]
        with pytest.raises(ValueError) as exc_info:
            fuse_all_results(results, method="invalid")
        expected = "Unsupported fusion method: invalid"
        result = str(exc_info.value)
        assert result == expected, f"Expected error message '{expected}', got '{result}'"
