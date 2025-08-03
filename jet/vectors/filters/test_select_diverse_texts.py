import pytest
import faiss
import numpy as np

from jet.logger import logger
from jet.vectors.filters import select_diverse_texts


class TestSelectDiverseTexts:
    """Tests for the select_diverse_texts function."""

    def test_select_diverse_texts_with_similar_and_dissimilar_texts(self):
        """Test selecting diverse texts when some texts are similar and others are dissimilar."""
        # Given: A set of texts and embeddings with some similar and some dissimilar texts
        cluster_texts = [
            "The cat is on the mat",
            "The cat sits on the mat",
            "The dog runs in the park",
            "Birds fly in the sky"
        ]
        # Mock embeddings: texts 0 and 1 are similar, texts 2 and 3 are dissimilar to 0,1 and each other
        cluster_embeddings = np.array([
            [0.9, 0.1],  # Text 0
            [0.9, 0.1],  # Text 1: similar to 0
            [0.1, 0.9],  # Text 2: dissimilar to 0,1
            [-0.9, 0.1]  # Text 3: dissimilar to 0,1,2
        ], dtype=np.float32)
        cluster_embeddings = np.ascontiguousarray(cluster_embeddings)
        logger.debug(
            f"cluster_embeddings dtype: {cluster_embeddings.dtype}, shape: {cluster_embeddings.shape}, is_contiguous: {cluster_embeddings.flags['C_CONTIGUOUS']}")
        faiss.normalize_L2(cluster_embeddings)
        logger.debug(
            f"After normalization, cluster_embeddings: {cluster_embeddings}")
        initial_text_idx = 0
        diversity_threshold = 0.8
        expected = [
            "The cat is on the mat",
            "The dog runs in the park",
            "Birds fly in the sky"
        ]

        # When: Selecting diverse texts
        result = select_diverse_texts(
            cluster_embeddings=cluster_embeddings,
            cluster_texts=cluster_texts,
            initial_text_idx=initial_text_idx,
            diversity_threshold=diversity_threshold,
            max_diverse_texts=3
        )

        # Then: Expect the initial text plus two dissimilar texts
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_select_diverse_texts_with_all_similar_texts(self):
        """Test selecting diverse texts when all texts are very similar."""
        # Given: A set of texts that are all very similar
        cluster_texts = [
            "The cat is on the mat",
            "The cat sits on the mat",
            "The cat lies on the mat"
        ]
        # Mock embeddings: all texts are very similar
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
        expected = ["The cat is on the mat"]

        # When: Selecting diverse texts
        result = select_diverse_texts(
            cluster_embeddings=cluster_embeddings,
            cluster_texts=cluster_texts,
            initial_text_idx=initial_text_idx,
            diversity_threshold=diversity_threshold,
            max_diverse_texts=3
        )

        # Then: Expect only the initial text due to high similarity
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_select_diverse_texts_with_empty_input(self):
        """Test selecting diverse texts with empty inputs."""
        # Given: Empty texts and embeddings
        cluster_texts = []
        cluster_embeddings = np.array([])
        initial_text_idx = 0
        diversity_threshold = 0.8
        expected = []

        # When: Selecting diverse texts
        result = select_diverse_texts(
            cluster_embeddings=cluster_embeddings,
            cluster_texts=cluster_texts,
            initial_text_idx=initial_text_idx,
            diversity_threshold=diversity_threshold,
            max_diverse_texts=3
        )

        # Then: Expect empty result
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_select_diverse_texts_with_invalid_initial_index(self):
        """Test selecting diverse texts with an invalid initial index."""
        # Given: Valid texts and embeddings but invalid initial index
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
        initial_text_idx = 5  # Out of bounds
        diversity_threshold = 0.8
        expected = []

        # When: Selecting diverse texts
        result = select_diverse_texts(
            cluster_embeddings=cluster_embeddings,
            cluster_texts=cluster_texts,
            initial_text_idx=initial_text_idx,
            diversity_threshold=diversity_threshold,
            max_diverse_texts=3
        )

        # Then: Expect empty result due to invalid index
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_select_diverse_texts_with_low_diversity_threshold(self):
        """Test selecting diverse texts with a low diversity threshold."""
        # Given: Texts and embeddings with a low diversity threshold
        cluster_texts = [
            "The cat is on the mat",
            "The dog runs in the park",
            "Birds fly in the sky"
        ]
        # Mock embeddings: all texts are somewhat similar to ensure rejection at low threshold
        cluster_embeddings = np.array([
            [0.9, 0.1],   # Text 0
            [0.85, 0.15],  # Text 1: similar to 0
            [0.8, 0.2]   # Text 2: similar to 0
        ], dtype=np.float32)
        cluster_embeddings = np.ascontiguousarray(cluster_embeddings)
        logger.debug(
            f"cluster_embeddings dtype: {cluster_embeddings.dtype}, shape: {cluster_embeddings.shape}, is_contiguous: {cluster_embeddings.flags['C_CONTIGUOUS']}")
        faiss.normalize_L2(cluster_embeddings)
        logger.debug(
            f"After normalization, cluster_embeddings: {cluster_embeddings}")
        initial_text_idx = 0
        diversity_threshold = 0.2  # Very low, so only initial text is selected
        expected = ["The cat is on the mat"]

        # When: Selecting diverse texts
        result = select_diverse_texts(
            cluster_embeddings=cluster_embeddings,
            cluster_texts=cluster_texts,
            initial_text_idx=initial_text_idx,
            diversity_threshold=diversity_threshold,
            max_diverse_texts=3
        )

        # Then: Expect only the initial text due to low threshold
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_select_diverse_texts_with_dynamic_max_diverse_texts(self):
        """Test selecting diverse texts with dynamic max_diverse_texts."""
        # Given: A set of texts and embeddings with some dissimilar texts
        cluster_texts = [
            "The cat is on the mat",
            "The cat sits on the mat",
            "The dog runs in the park",
            "Birds fly in the sky",
            "Fish swim in the ocean"
        ]
        # Mock embeddings: texts 0,1 similar; 2,3,4 dissimilar to 0,1 and each other
        cluster_embeddings = np.array([
            [0.9, 0.1],   # Text 0
            [0.9, 0.1],   # Text 1: similar to 0
            [0.1, 0.9],   # Text 2: dissimilar to 0,1
            [-0.9, 0.1],  # Text 3: dissimilar to 0,1,2
            [0.1, -0.9]   # Text 4: dissimilar to 0,1,2,3
        ], dtype=np.float32)
        cluster_embeddings = np.ascontiguousarray(cluster_embeddings)
        logger.debug(
            f"cluster_embeddings dtype: {cluster_embeddings.dtype}, shape: {cluster_embeddings.shape}, is_contiguous: {cluster_embeddings.flags['C_CONTIGUOUS']}")
        faiss.normalize_L2(cluster_embeddings)
        logger.debug(
            f"After normalization, cluster_embeddings: {cluster_embeddings}")
        initial_text_idx = 0
        diversity_threshold = 0.8
        expected = [
            "The cat is on the mat",
            "The dog runs in the park",
            "Birds fly in the sky"
        ]

        # When: Selecting diverse texts with max_diverse_texts=None
        result = select_diverse_texts(
            cluster_embeddings=cluster_embeddings,
            cluster_texts=cluster_texts,
            initial_text_idx=initial_text_idx,
            diversity_threshold=diversity_threshold,
            max_diverse_texts=None
        )

        # Then: Expect up to 3 diverse texts (dynamic default)
        assert result == expected, f"Expected {expected}, but got {result}"
