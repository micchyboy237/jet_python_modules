import pytest
import numpy as np
from jet.llm.rag.mlx.classification import MLXRAGClassifier
from jet.logger import logger
from typing import List
from collections import defaultdict


class TestMLXRAGClassifier:
    """Tests for MLXRAGClassifier ensuring batches do not mix group_ids."""

    @pytest.fixture
    def classifier(self) -> MLXRAGClassifier:
        """Fixture to initialize MLXRAGClassifier with small batch size."""
        return MLXRAGClassifier(model_name="qwen3-1.7b-4bit", batch_size=2, show_progress=False)

    def test_generate_embeddings_no_mixed_group_ids(self, classifier: MLXRAGClassifier):
        """Test that batches do not contain mixed group_ids."""
        # Given
        chunks = ["chunk1", "chunk2", "chunk3", "chunk4", "chunk5"]
        group_ids = ["group1", "group1", "group2", "group2", "group3"]
        expected_batch_groups = [
            ["chunk1", "chunk2"],  # group1
            ["chunk3", "chunk4"],  # group2
            ["chunk5"],           # group3
        ]
        expected_chunk_count = len(chunks)

        # When
        embeddings = classifier.generate_embeddings(
            chunks, group_ids=group_ids)

        # Then
        result_batches = []
        for i in range(0, len(chunks), classifier.batch_size):
            batch_chunks = chunks[i:i + classifier.batch_size]
            if batch_chunks:
                result_batches.append(batch_chunks)
        result_chunk_count = embeddings.shape[0]

        assert result_batches == expected_batch_groups, f"Expected batches {expected_batch_groups}, got {result_batches}"
        assert result_chunk_count == expected_chunk_count, f"Expected {expected_chunk_count} embeddings, got {result_chunk_count}"
        # Verify no mixed group_ids in batches
        for batch in result_batches:
            batch_indices = [chunks.index(chunk) for chunk in batch]
            batch_groups = [group_ids[idx] for idx in batch_indices]
            assert len(
                set(batch_groups)) == 1, f"Batch {batch} contains mixed groups: {batch_groups}"

    def test_generate_embeddings_fallback_no_group_ids(self, classifier: MLXRAGClassifier):
        """Test fallback batching when group_ids is not provided."""
        # Given
        chunks = ["chunk1", "chunk2", "chunk3", "chunk4"]
        expected_batch_groups = [
            ["chunk1", "chunk2"],
            ["chunk3", "chunk4"],
        ]
        expected_chunk_count = len(chunks)

        # When
        embeddings = classifier.generate_embeddings(chunks)

        # Then
        result_batches = []
        for i in range(0, len(chunks), classifier.batch_size):
            batch_chunks = chunks[i:i + classifier.batch_size]
            if batch_chunks:
                result_batches.append(batch_chunks)
        result_chunk_count = embeddings.shape[0]

        assert result_batches == expected_batch_groups, f"Expected batches {expected_batch_groups}, got {result_batches}"
        assert result_chunk_count == expected_chunk_count, f"Expected {expected_chunk_count} embeddings, got {result_chunk_count}"
