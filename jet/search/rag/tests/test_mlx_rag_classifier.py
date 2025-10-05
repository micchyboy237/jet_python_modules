import pytest
import numpy as np
from jet.llm.rag.mlx.classification import MLXRAGClassifier, _model_cache, _tokenizer_cache, ClassificationResult
from jet.logger import logger
from typing import List, Literal
from collections import defaultdict
import time
import uuid


class TestMLXRAGClassifier:
    """Tests for MLXRAGClassifier ensuring batch grouping and performance."""

    def test_model_and_tokenizer_caching(self):
        model_name = "qwen3-1.7b-4bit"
        _model_cache.clear()
        _tokenizer_cache.clear()
        classifier1 = MLXRAGClassifier(model_name=model_name)
        classifier2 = MLXRAGClassifier(model_name=model_name)
        expected_model = classifier1.model
        expected_tokenizer = classifier1.tokenizer
        result_model = classifier2.model
        result_tokenizer = classifier2.tokenizer
        assert result_model is expected_model, "Model instances should be identical (cached)"
        assert result_tokenizer is expected_tokenizer, "Tokenizer instances should be identical (cached)"
        assert model_name in _model_cache, "Model should be in cache"
        assert model_name in _tokenizer_cache, "Tokenizer should be in cache"

    @pytest.fixture
    def classifier(self) -> MLXRAGClassifier:
        """Fixture to initialize MLXRAGClassifier with small batch size."""
        return MLXRAGClassifier(model_name="qwen3-1.7b-4bit", batch_size=2, show_progress=False)

    def test_generate_embeddings_no_mixed_group_ids(self, classifier: MLXRAGClassifier):
        """Test that batches do not contain mixed group_ids."""
        chunks = ["chunk1", "chunk2", "chunk3", "chunk4", "chunk5"]
        group_ids = ["group1", "group1", "group2", "group2", "group3"]
        expected_batch_groups = [
            ["chunk1", "chunk2"],
            ["chunk3", "chunk4"],
            ["chunk5"],
        ]
        expected_chunk_count = len(chunks)
        embeddings = classifier.generate_embeddings(
            chunks, group_ids=group_ids)
        result_batches = []
        batch_size = classifier.batch_size
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            if batch_chunks:
                result_batches.append(batch_chunks)
        result_chunk_count = embeddings.shape[0]
        assert result_batches == expected_batch_groups, f"Expected batches {expected_batch_groups}, got {result_batches}"
        assert result_chunk_count == expected_chunk_count, f"Expected {expected_chunk_count} embeddings, got {result_chunk_count}"
        for batch in result_batches:
            batch_indices = [chunks.index(chunk) for chunk in batch]
            batch_groups = [group_ids[idx] for idx in batch_indices]
            assert len(
                set(batch_groups)) == 1, f"Batch {batch} contains mixed groups: {batch_groups}"

    def test_generate_embeddings_fallback_no_group_ids(self, classifier: MLXRAGClassifier):
        """Test fallback batching when group_ids is not provided."""
        chunks = ["chunk1", "chunk2", "chunk3", "chunk4"]
        expected_batch_groups = [
            ["chunk1", "chunk2"],
            ["chunk3", "chunk4"],
        ]
        expected_chunk_count = len(chunks)
        embeddings = classifier.generate_embeddings(chunks)
        result_batches = []
        batch_size = classifier.batch_size
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            if batch_chunks:
                result_batches.append(batch_chunks)
        result_chunk_count = embeddings.shape[0]
        assert result_batches == expected_batch_groups, f"Expected batches {expected_batch_groups}, got {result_batches}"
        assert result_chunk_count == expected_chunk_count, f"Expected {expected_chunk_count} embeddings, got {result_chunk_count}"

    def test_generate_embeddings_performance(self, classifier: MLXRAGClassifier):
        """Test performance improvement for large chunk counts."""
        chunks = ["Sample text " * 20 for _ in range(100)]
        group_ids = ["group1"] * 50 + ["group2"] * 50
        expected_chunk_count = len(chunks)
        start_time = time.time()
        embeddings = classifier.generate_embeddings(
            chunks, group_ids=group_ids)
        elapsed_time = time.time() - start_time
        assert embeddings.shape[
            0] == expected_chunk_count, f"Expected {expected_chunk_count} embeddings, got {embeddings.shape[0]}"
        assert elapsed_time < 10.0, f"Embedding generation took too long: {elapsed_time:.2f} seconds"

    def test_generate_embeddings_accuracy_repeated_chunks(self, classifier: MLXRAGClassifier):
        """Test that repeated chunks produce identical embeddings via caching."""
        chunks = ["identical chunk"] * 5
        group_ids = ["group1"] * 5
        expected_chunk_count = len(chunks)
        expected_cache_hits = 4
        embeddings = classifier.generate_embeddings(
            chunks, group_ids=group_ids)
        assert embeddings.shape[
            0] == expected_chunk_count, f"Expected {expected_chunk_count} embeddings, got {embeddings.shape[0]}"
        for i in range(1, len(embeddings)):
            np.testing.assert_array_almost_equal(
                embeddings[0], embeddings[i], decimal=4,
                err_msg=f"Embeddings for identical chunks differ at index {i}"
            )
        chunk_hash = classifier._hash_text("identical chunk")
        assert chunk_hash in classifier.embedding_cache, "Chunk hash not found in cache"
        assert len(
            classifier.embedding_cache) == 1, f"Expected 1 unique chunk in cache, got {len(classifier.embedding_cache)}"
        assert classifier.embedding_cache[chunk_hash].shape == embeddings[0].shape, "Cached embedding shape mismatch"


class TestMLXRAGClassifierSort:
    @pytest.fixture
    def classifier(self):
        return MLXRAGClassifier(model_name="qwen3-1.7b-4bit", batch_size=2, show_progress=False)

    @pytest.fixture
    def sample_chunks(self):
        return [
            "This is a relevant chunk about AI.",
            "This is another relevant chunk about machine learning.",
            "This is a non-relevant chunk about cooking."
        ]

    @pytest.fixture
    def sample_embeddings(self, classifier, sample_chunks):
        return classifier.generate_embeddings(sample_chunks)

    def test_generate_selects_highest_score(self, classifier, sample_chunks, sample_embeddings):
        query = "AI and machine learning"
        expected_label: Literal["relevant", "non-relevant"] = "relevant"
        expected_threshold = 0.7
        result = classifier.generate(
            query, sample_chunks, sample_embeddings, relevance_threshold=expected_threshold)
        assert result == expected_label, f"Expected label {expected_label}, but got {result}"

    def test_stream_generate_sorts_by_score_descending(self, classifier, sample_chunks, sample_embeddings):
        query = "AI and machine learning"
        top_k = 3
        expected_threshold = 0.925
        expected_labels: List[Literal["relevant", "non-relevant"]
                              ] = ["relevant", "non-relevant", "non-relevant"]
        expected_scores = []
        results = list(classifier.stream_generate(query, sample_chunks,
                       sample_embeddings, top_k=top_k, relevance_threshold=expected_threshold))
        result_labels = [label for label, _, _ in results]
        result_scores = [score for _, score, _ in results]
        assert len(
            results) == top_k, f"Expected {top_k} results, but got {len(results)}"
        assert result_labels == expected_labels, f"Expected labels {expected_labels}, but got {result_labels}"
        assert all(result_scores[i] >= result_scores[i + 1] for i in range(len(result_scores) - 1)), \
            "Scores are not in descending order"


class TestMLXRAGClassifierClassify:
    @pytest.fixture
    def classifier(self) -> MLXRAGClassifier:
        return MLXRAGClassifier(model_name="qwen3-1.7b-4bit", batch_size=2, show_progress=False)

    @pytest.fixture
    def sample_chunks(self) -> List[str]:
        return [
            "AI is transforming industries.",
            "Machine learning improves efficiency.",
            "Cooking recipes are delicious."
        ]

    @pytest.fixture
    def sample_embeddings(self, classifier: MLXRAGClassifier, sample_chunks: List[str]) -> np.ndarray:
        return classifier.generate_embeddings(sample_chunks)

    def test_classify_with_provided_ids(self, classifier: MLXRAGClassifier, sample_chunks: List[str], sample_embeddings: np.ndarray):
        # Given a query, chunks, embeddings, and custom IDs
        query = "AI and machine learning"
        custom_ids = ["id1", "id2", "id3"]
        # Expected results ordered by score (highest first), based on test logs
        expected_results: List[ClassificationResult] = [
            {
                "id": "id3",
                "doc_index": 2,
                "rank": 1,
                "score": 0.913122296333313,
                "text": "Cooking recipes are delicious.",
                "label": "relevant",
                "threshold": 0.7
            },
            {
                "id": "id2",
                "doc_index": 1,
                "rank": 2,
                "score": 0.8591018319129944,
                "text": "Machine learning improves efficiency.",
                "label": "relevant",
                "threshold": 0.7
            },
            {
                "id": "id1",
                "doc_index": 0,
                "rank": 3,
                "score": 0.5855764150619507,
                "text": "AI is transforming industries.",
                "label": "non-relevant",
                "threshold": 0.7
            }
        ]

        # When classifying with provided IDs
        result = classifier.classify(
            query, sample_chunks, sample_embeddings, ids=custom_ids, verbose=False)

        # Then results should match expected IDs, indices, texts, and be sorted by score
        assert len(result) == len(
            expected_results), f"Expected {len(expected_results)} results, got {len(result)}"
        for res, exp in zip(result, expected_results):
            assert res["id"] == exp["id"], f"Expected id {exp['id']}, got {res['id']}"
            assert res["doc_index"] == exp[
                "doc_index"], f"Expected doc_index {exp['doc_index']}, got {res['doc_index']}"
            assert res["text"] == exp["text"], f"Expected text {exp['text']}, got {res['text']}"
            assert res["rank"] == exp["rank"], f"Expected rank {exp['rank']}, got {res['rank']}"
            assert abs(res["score"] - exp["score"]
                       ) < 0.0001, f"Expected score {exp['score']}, got {res['score']}"
        assert all(result[i]["score"] >= result[i + 1]["score"]
                   for i in range(len(result) - 1)), "Results not sorted by score"

    def test_classify_without_ids_generates_uuids(self, classifier: MLXRAGClassifier, sample_chunks: List[str], sample_embeddings: np.ndarray):
        # Given a query, chunks, and embeddings without IDs
        query = "AI and machine learning"
        expected_count = len(sample_chunks)
        # Expected indices based on sorted order (highest score first)
        expected_indices = [2, 1, 0]

        # When classifying without IDs
        result = classifier.classify(
            query, sample_chunks, sample_embeddings, verbose=False)

        # Then UUIDs are generated, and results match chunks in sorted order
        assert len(
            result) == expected_count, f"Expected {expected_count} results, got {len(result)}"
        for res, chunk, idx in zip(result, [sample_chunks[i] for i in expected_indices], expected_indices):
            assert isinstance(
                uuid.UUID(res["id"]), uuid.UUID), f"Expected UUID for id, got {res['id']}"
            assert res["doc_index"] == idx, f"Expected doc_index {idx}, got {res['doc_index']}"
            assert res["text"] == chunk, f"Expected text {chunk}, got {res['text']}"
            assert res["rank"] > 0, f"Expected rank > 0, got {res['rank']}"
        assert all(result[i]["score"] >= result[i + 1]["score"]
                   for i in range(len(result) - 1)), "Results not sorted by score"

    def test_classify_with_mismatched_ids_fallback_to_uuids(self, classifier: MLXRAGClassifier, sample_chunks: List[str], sample_embeddings: np.ndarray):
        # Given a query, chunks, embeddings, and mismatched IDs
        query = "AI and machine learning"
        mismatched_ids = ["id1", "id2"]
        expected_count = len(sample_chunks)
        # Expected indices based on sorted order
        expected_indices = [2, 1, 0]

        # When classifying with mismatched IDs
        result = classifier.classify(
            query, sample_chunks, sample_embeddings, ids=mismatched_ids, verbose=False)

        # Then UUIDs are generated, and results match chunks in sorted order
        assert len(
            result) == expected_count, f"Expected {expected_count} results, got {len(result)}"
        for res, chunk, idx in zip(result, [sample_chunks[i] for i in expected_indices], expected_indices):
            assert isinstance(
                uuid.UUID(res["id"]), uuid.UUID), f"Expected UUID for id, got {res['id']}"
            assert res["doc_index"] == idx, f"Expected doc_index {idx}, got {res['doc_index']}"
            assert res["text"] == chunk, f"Expected text {chunk}, got {res['text']}"
            assert res["rank"] > 0, f"Expected rank > 0, got {res['rank']}"
        assert all(result[i]["score"] >= result[i + 1]["score"]
                   for i in range(len(result) - 1)), "Results not sorted by score"

    def test_classify_verbose_logging(self, classifier: MLXRAGClassifier, sample_chunks: List[str], sample_embeddings: np.ndarray):
        # Given a query, chunks, embeddings, and verbose=True
        query = "AI and machine learning"
        expected_results: List[ClassificationResult] = [
            {
                "id": "94b0872b-c154-4667-a784-2f0e0b19abad",
                "doc_index": 2,
                "rank": 1,
                "score": 0.913122296333313,
                "text": "Cooking recipes are delicious.",
                "label": "relevant",
                "threshold": 0.7
            },
            {
                "id": "99baf3e1-e258-4af6-b64b-3345f0f2182e",
                "doc_index": 1,
                "rank": 2,
                "score": 0.8591018319129944,
                "text": "Machine learning improves efficiency.",
                "label": "relevant",
                "threshold": 0.7
            },
            {
                "id": "ef0d381d-264f-4e07-9eb1-7f2b628fcc08",
                "doc_index": 0,
                "rank": 3,
                "score": 0.5855764150619507,
                "text": "AI is transforming industries.",
                "label": "non-relevant",
                "threshold": 0.7
            }
        ]

        # When classifying with verbose logging enabled
        result = classifier.classify(
            query, sample_chunks, sample_embeddings, verbose=True)

        # Then results should match expected structure and be sorted by score
        assert len(result) == len(
            expected_results), f"Expected {len(expected_results)} results, got {len(result)}"
        for res, exp in zip(result, expected_results):
            assert isinstance(
                uuid.UUID(res["id"]), uuid.UUID), f"Expected UUID for id, got {res['id']}"
            assert res["doc_index"] == exp[
                "doc_index"], f"Expected doc_index {exp['doc_index']}, got {res['doc_index']}"
            assert res["text"] == exp["text"], f"Expected text {exp['text']}, got {res['text']}"
            assert res["rank"] == exp["rank"], f"Expected rank {exp['rank']}, got {res['rank']}"
            assert abs(res["score"] - exp["score"]
                       ) < 0.0001, f"Expected score {exp['score']}, got {res['score']}"
        assert all(result[i]["score"] >= result[i + 1]["score"]
                   for i in range(len(result) - 1)), "Results not sorted by score"

    def test_classify_empty_chunks(self, classifier: MLXRAGClassifier):
        # Given an empty chunk list and empty embeddings
        query = "AI and machine learning"
        chunks: List[str] = []
        embeddings = np.array([])

        # When classifying
        result = classifier.classify(query, chunks, embeddings)

        # Then an empty list is returned
        expected_results: List[ClassificationResult] = []
        assert result == expected_results, f"Expected empty results, got {result}"
