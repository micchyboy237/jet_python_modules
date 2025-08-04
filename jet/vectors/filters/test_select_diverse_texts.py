import pytest
import faiss
import numpy as np
from jet.logger import logger
from jet.vectors.filters import select_diverse_texts, calculate_max_diverse_texts
from typing import List
import uuid


class TestSelectDiverseTexts:
    """Tests for the select_diverse_texts function."""

    def test_select_diverse_texts_with_similar_and_dissimilar_texts(self):
        """Test selecting diverse texts when some texts are similar and others are dissimilar."""
        # Given
        cluster_texts = [
            "The cat is on the mat",
            "The cat sits on the mat",
            "The dog runs in the park",
            "Birds fly in the sky"
        ]
        cluster_embeddings = np.array([
            [0.9, 0.1],
            [0.9, 0.1],
            [0.1, 0.9],
            [-0.9, 0.1]
        ], dtype=np.float32)
        cluster_embeddings = np.ascontiguousarray(cluster_embeddings)
        logger.debug(
            f"cluster_embeddings dtype: {cluster_embeddings.dtype}, shape: {cluster_embeddings.shape}, is_contiguous: {cluster_embeddings.flags['C_CONTIGUOUS']}")
        faiss.normalize_L2(cluster_embeddings)
        logger.debug(
            f"After normalization, cluster_embeddings: {cluster_embeddings}")
        initial_text_idx = 0
        diversity_threshold = 0.8
        ids = [str(uuid.uuid4()) for _ in range(len(cluster_texts))]
        expected = [
            {"id": ids[0], "index": 0,
                "text": "The cat is on the mat", "score": 1.0},
            {"id": ids[2], "index": 2, "text": "The dog runs in the park",
                "score": pytest.approx(0.2195122, abs=1e-4)},
            {"id": ids[3], "index": 3, "text": "Birds fly in the sky",
                "score": pytest.approx(-0.9756098, abs=1e-4)}
        ]

        # When
        result = select_diverse_texts(
            cluster_embeddings=cluster_embeddings,
            cluster_texts=cluster_texts,
            initial_text_idx=initial_text_idx,
            diversity_threshold=diversity_threshold,
            max_diverse_texts=3,
            ids=ids
        )

        # Then
        assert len(result) == len(
            expected), f"Expected {len(expected)} results, but got {len(result)}"
        for r, e in zip(result, expected):
            assert r["id"] == e["id"], f"Expected id {e['id']}, but got {r['id']}"
            assert r["index"] == e["index"], f"Expected index {e['index']}, but got {r['index']}"
            assert r["text"] == e["text"], f"Expected text {e['text']}, but got {r['text']}"
            assert r["score"] == pytest.approx(
                e["score"], abs=1e-4), f"Expected score {e['score']}, but got {r['score']}"

    def test_select_diverse_texts_with_all_similar_texts(self):
        """Test selecting diverse texts when all texts are very similar."""
        # Given
        cluster_texts = [
            "The cat is on the mat",
            "The cat sits on the mat",
            "The cat lies on the mat"
        ]
        cluster_embeddings = np.array([
            [0.9, 0.1],
            [0.91, 0.1],
            [0.89, 0.11]
        ], dtype=np.float32)
        cluster_embeddings = np.ascontiguousarray(cluster_embeddings)
        logger.debug(
            f"cluster_embeddings dtype: {cluster_embeddings.dtype}, shape: {cluster_embeddings.shape}, is_contiguous: {cluster_embeddings.flags['C_CONTIGUOUS']}")
        faiss.normalize_L2(cluster_embeddings)
        logger.debug(
            f"After normalization, cluster_embeddings: {cluster_embeddings}")
        initial_text_idx = 0
        diversity_threshold = 0.8
        ids = [str(uuid.uuid4()) for _ in range(len(cluster_texts))]
        expected = [{"id": ids[0], "index": 0,
                     "text": "The cat is on the mat", "score": 1.0}]

        # When
        result = select_diverse_texts(
            cluster_embeddings=cluster_embeddings,
            cluster_texts=cluster_texts,
            initial_text_idx=initial_text_idx,
            diversity_threshold=diversity_threshold,
            max_diverse_texts=3,
            ids=ids
        )

        # Then
        assert len(result) == len(
            expected), f"Expected {len(expected)} results, but got {len(result)}"
        for r, e in zip(result, expected):
            assert r["id"] == e["id"], f"Expected id {e['id']}, but got {r['id']}"
            assert r["index"] == e["index"], f"Expected index {e['index']}, but got {r['index']}"
            assert r["text"] == e["text"], f"Expected text {e['text']}, but got {r['text']}"
            assert r["score"] == pytest.approx(
                e["score"], abs=1e-4), f"Expected score {e['score']}, but got {r['score']}"

    def test_select_diverse_texts_with_empty_input(self):
        """Test selecting diverse texts with empty inputs."""
        # Given
        cluster_texts = []
        cluster_embeddings = np.array([])
        initial_text_idx = 0
        diversity_threshold = 0.8
        expected = []

        # When
        result = select_diverse_texts(
            cluster_embeddings=cluster_embeddings,
            cluster_texts=cluster_texts,
            initial_text_idx=initial_text_idx,
            diversity_threshold=diversity_threshold,
            max_diverse_texts=3
        )

        # Then
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_select_diverse_texts_with_invalid_initial_index(self):
        """Test selecting diverse texts with an invalid initial index."""
        # Given
        cluster_texts = [
            "The cat is on the mat",
            "The dog runs in the park"
        ]
        cluster_embeddings = np.array([
            [0.9, 0.1],
            [0.1, 0.9]
        ], dtype=np.float32)
        cluster_embeddings = np.ascontiguousarray(cluster_embeddings)
        logger.debug(
            f"cluster_embeddings dtype: {cluster_embeddings.dtype}, shape: {cluster_embeddings.shape}, is_contiguous: {cluster_embeddings.flags['C_CONTIGUOUS']}")
        faiss.normalize_L2(cluster_embeddings)
        logger.debug(
            f"After normalization, cluster_embeddings: {cluster_embeddings}")
        initial_text_idx = 5
        diversity_threshold = 0.8
        expected = []

        # When
        result = select_diverse_texts(
            cluster_embeddings=cluster_embeddings,
            cluster_texts=cluster_texts,
            initial_text_idx=initial_text_idx,
            diversity_threshold=diversity_threshold,
            max_diverse_texts=3
        )

        # Then
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_select_diverse_texts_with_low_diversity_threshold(self):
        """Test selecting diverse texts with a low diversity threshold."""
        # Given
        cluster_texts = [
            "The cat is on the mat",
            "The dog runs in the park",
            "Birds fly in the sky"
        ]
        cluster_embeddings = np.array([
            [0.9, 0.1],
            [0.85, 0.15],
            [0.8, 0.2]
        ], dtype=np.float32)
        cluster_embeddings = np.ascontiguousarray(cluster_embeddings)
        logger.debug(
            f"cluster_embeddings dtype: {cluster_embeddings.dtype}, shape: {cluster_embeddings.shape}, is_contiguous: {cluster_embeddings.flags['C_CONTIGUOUS']}")
        faiss.normalize_L2(cluster_embeddings)
        logger.debug(
            f"After normalization, cluster_embeddings: {cluster_embeddings}")
        initial_text_idx = 0
        diversity_threshold = 0.2
        ids = [str(uuid.uuid4()) for _ in range(len(cluster_texts))]
        expected = [{"id": ids[0], "index": 0,
                     "text": "The cat is on the mat", "score": 1.0}]

        # When
        result = select_diverse_texts(
            cluster_embeddings=cluster_embeddings,
            cluster_texts=cluster_texts,
            initial_text_idx=initial_text_idx,
            diversity_threshold=diversity_threshold,
            max_diverse_texts=3,
            ids=ids
        )

        # Then
        assert len(result) == len(
            expected), f"Expected {len(expected)} results, but got {len(result)}"
        for r, e in zip(result, expected):
            assert r["id"] == e["id"], f"Expected id {e['id']}, but got {r['id']}"
            assert r["index"] == e["index"], f"Expected index {e['index']}, but got {r['index']}"
            assert r["text"] == e["text"], f"Expected text {e['text']}, but got {r['text']}"
            assert r["score"] == pytest.approx(
                e["score"], abs=1e-4), f"Expected score {e['score']}, but got {r['score']}"

    def test_select_diverse_texts_with_dynamic_max_diverse_texts(self):
        """Test selecting diverse texts with dynamic max_diverse_texts."""
        # Given
        cluster_texts = [
            "The cat is on the mat",
            "The cat sits on the mat",
            "The dog runs in the park",
            "Birds fly in the sky",
            "Fish swim in the ocean",
            "Horses gallop in the field"
        ]
        cluster_embeddings = np.array([
            [0.9, 0.1],
            [0.9, 0.1],
            [0.1, 0.9],
            [-0.9, 0.1],
            [0.1, -0.9],
            [-0.1, -0.9]
        ], dtype=np.float32)
        cluster_embeddings = np.ascontiguousarray(cluster_embeddings)
        logger.debug(
            f"cluster_embeddings dtype: {cluster_embeddings.dtype}, shape: {cluster_embeddings.shape}, is_contiguous: {cluster_embeddings.flags['C_CONTIGUOUS']}")
        faiss.normalize_L2(cluster_embeddings)
        logger.debug(
            f"After normalization, cluster_embeddings: {cluster_embeddings}")
        initial_text_idx = 0
        diversity_threshold = 0.8
        ids = [str(uuid.uuid4()) for _ in range(len(cluster_texts))]
        expected = [
            {"id": ids[0], "index": 0,
                "text": "The cat is on the mat", "score": 1.0},
            {"id": ids[2], "index": 2, "text": "The dog runs in the park",
                "score": pytest.approx(0.2195122, abs=1e-4)},
            {"id": ids[3], "index": 3, "text": "Birds fly in the sky",
                "score": pytest.approx(-0.9756098, abs=1e-4)},
            {"id": ids[4], "index": 4, "text": "Fish swim in the ocean",
                "score": pytest.approx(0.0, abs=1e-4)}
        ]
        expected_max_texts = calculate_max_diverse_texts(
            cluster_embeddings, cluster_texts)
        logger.debug(f"Expected max_diverse_texts: {expected_max_texts}")

        # When
        result = select_diverse_texts(
            cluster_embeddings=cluster_embeddings,
            cluster_texts=cluster_texts,
            initial_text_idx=initial_text_idx,
            diversity_threshold=diversity_threshold,
            max_diverse_texts=None,
            ids=ids
        )

        # Then
        assert len(result) == len(
            expected), f"Expected {len(expected)} results, but got {len(result)}"
        assert len(
            result) <= expected_max_texts, f"Result length {len(result)} exceeds expected max {expected_max_texts}"
        for r, e in zip(result, expected):
            assert r["id"] == e["id"], f"Expected id {e['id']}, but got {r['id']}"
            assert r["index"] == e["index"], f"Expected index {e['index']}, but got {r['index']}"
            assert r["text"] == e["text"], f"Expected text {e['text']}, but got {r['text']}"
            assert r["score"] == pytest.approx(
                e["score"], abs=1e-4), f"Expected score {e['score']}, but got {r['score']}"

    def test_select_diverse_texts_with_default_ids(self):
        """Test selecting diverse texts with default-generated UUIDs."""
        # Given
        cluster_texts = [
            "The cat is on the mat",
            "The dog runs in the park",
            "Birds fly in the sky"
        ]
        cluster_embeddings = np.array([
            [0.9, 0.1],
            [0.1, 0.9],
            [-0.9, 0.1]
        ], dtype=np.float32)
        cluster_embeddings = np.ascontiguousarray(cluster_embeddings)
        logger.debug(
            f"cluster_embeddings dtype: {cluster_embeddings.dtype}, shape: {cluster_embeddings.shape}, is_contiguous: {cluster_embeddings.flags['C_CONTIGUOUS']}")
        faiss.normalize_L2(cluster_embeddings)
        logger.debug(
            f"After normalization, cluster_embeddings: {cluster_embeddings}")
        initial_text_idx = 0
        diversity_threshold = 0.8
        expected_texts = [
            "The cat is on the mat",
            "The dog runs in the park",
            "Birds fly in the sky"
        ]

        # When
        result = select_diverse_texts(
            cluster_embeddings=cluster_embeddings,
            cluster_texts=cluster_texts,
            initial_text_idx=initial_text_idx,
            diversity_threshold=diversity_threshold,
            max_diverse_texts=3
        )

        # Then
        assert len(result) == len(
            expected_texts), f"Expected {len(expected_texts)} results, but got {len(result)}"
        for r, text in zip(result, expected_texts):
            assert isinstance(
                r["id"], str), f"Expected string ID, but got {type(r['id'])}"
            assert len(
                r["id"]) == 36, f"Expected UUID length 36, but got {len(r['id'])}"
            assert r["text"] == text, f"Expected text {text}, but got {r['text']}"
            assert isinstance(
                r["index"], int), f"Expected integer index, but got {type(r['index'])}"
            assert isinstance(
                r["score"], float), f"Expected float score, but got {type(r['score'])}"
