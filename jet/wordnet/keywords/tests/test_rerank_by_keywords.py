from typing import List, Optional, Tuple, Union
from unittest import mock
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from jet.wordnet.keywords.keyword_extraction import rerank_by_keywords, SimilarityResult
from sklearn.feature_extraction.text import CountVectorizer


@pytest.fixture
def mock_dependencies():
    """Mock external dependencies for rerank_by_keywords."""
    with patch("jet.wordnet.keywords.keyword_extraction.spacy.load") as mock_spacy, \
            patch("jet.wordnet.keywords.keyword_extraction.setup_keybert") as mock_setup_keybert, \
            patch("jet.wordnet.keywords.keyword_extraction.SentenceTransformerRegistry.load_model") as mock_load_model, \
            patch("jet.wordnet.keywords.keyword_extraction.generate_embeddings") as mock_generate_embeddings, \
            patch("jet.wordnet.keywords.keyword_extraction.preprocess_text_for_rag") as mock_preprocess, \
            patch("jet.wordnet.keywords.keyword_extraction._count_tokens") as mock_count_tokens:

        mock_nlp = MagicMock()
        mock_spacy.return_value = mock_nlp

        mock_keybert = MagicMock()
        mock_setup_keybert.return_value = mock_keybert

        mock_embed_model = MagicMock()
        mock_load_model.return_value = mock_embed_model

        mock_generate_embeddings.side_effect = lambda texts, * \
            args, **kwargs: np.array([[0.1 * i] for i in range(len(texts))])

        mock_preprocess.side_effect = lambda x: x.lower()

        mock_count_tokens.side_effect = lambda text, nlp: len(text.split())

        yield mock_keybert, mock_embed_model, mock_nlp


class TestRerankByKeywordsBasic:
    def test_rerank_with_list_of_strings(self, mock_dependencies):
        # Given: A list of three texts and seed keywords
        mock_keybert, _, _ = mock_dependencies
        texts = [
            "The quick brown fox jumps over the lazy dog",
            "A fast fox leaps over obstacles",
            "The dog sleeps by the fire"
        ]
        seed_keywords = ["fox", "dog"]
        expected_keywords = [
            [("quick fox", 0.8), ("lazy dog", 0.7)],
            [("fast fox", 0.9)],
            [("dog sleeps", 0.6)]
        ]
        mock_keybert.extract_keywords.return_value = expected_keywords
        expected_result_length = len(texts)
        expected_scores = [0.9, 0.8, 0.6]

        # When: rerank_by_keywords is called with a list of strings
        results = rerank_by_keywords(
            texts=texts,
            seed_keywords=seed_keywords,
            top_n=2,
            show_progress=False
        )

        # Then: The result length matches the input length and scores are correct
        assert len(
            results) == expected_result_length, f"Expected {expected_result_length} results, got {len(results)}"
        assert [r["score"]
                for r in results] == expected_scores, f"Expected scores {expected_scores}, got {[r['score'] for r in results]}"
        assert all(r["rank"] == i + 1 for i, r in enumerate(results)
                   ), "Ranks should be sequential starting from 1"
        assert all(r["text"] == texts[r["doc_index"]]
                   for r in results), "Text should match input text"
        assert all(len(r["keywords"]) <=
                   2 for r in results), "Keywords should respect top_n=2"

    def test_rerank_with_matrix_of_texts(self, mock_dependencies):
        # Given: A matrix of two documents, each with two texts, and seed keywords
        mock_keybert, _, _ = mock_dependencies
        texts = [
            [
                "The quick brown fox jumps over the lazy dog",
                "Foxes are clever animals"
            ],
            [
                "A fast fox leaps over obstacles",
                "The forest is home to swift creatures"
            ]
        ]
        seed_keywords = ["fox", "dog"]
        expected_keywords = [
            [("quick fox", 0.8), ("lazy dog", 0.7)],  # For text1
            [("clever foxes", 0.6)],                  # For text2
            [("fast fox", 0.9)],                      # For text3
            [("swift creatures", 0.5)]                # For text4
        ]
        mock_keybert.extract_keywords.return_value = expected_keywords
        expected_result_length = len(texts)
        expected_scores = [0.7, 0.45]

        # When: rerank_by_keywords is called with a matrix of texts
        results = rerank_by_keywords(
            texts=texts,
            seed_keywords=seed_keywords,
            top_n=2,
            show_progress=False
        )

        # Then: The result length matches the input length and scores are averaged correctly
        assert len(
            results) == expected_result_length, f"Expected {expected_result_length} results, got {len(results)}"
        assert [round(r["score"], 4) for r in results] == [round(s, 4) for s in expected_scores], \
            f"Expected scores {[round(s, 4) for s in expected_scores]}, got {[round(r['score'], 4) for r in results]}"
        assert all(r["rank"] == i + 1 for i, r in enumerate(results)
                   ), "Ranks should be sequential starting from 1"
        assert all(r["text"] == " ".join(texts[r["doc_index"]])
                   for r in results), "Text should be joined input texts"
        assert all(len(r["keywords"]) <=
                   2 for r in results), "Keywords should respect top_n=2"
        # Additional assertion for keyword content
        assert all(any(kw["text"] in [k[0] for sublist in expected_keywords[i*2:(i+1)*2] for k in sublist]
                       for kw in r["keywords"]) for i, r in enumerate(results)), "Keywords should match expected"

    def test_matrix_with_unequal_lengths_raises_error(self, mock_dependencies):
        # Given: A matrix with unequal inner list lengths
        mock_keybert, _, _ = mock_dependencies
        texts = [
            ["The quick brown fox", "Foxes are clever"],
            ["A fast fox"]  # Unequal length
        ]

        # When: rerank_by_keywords is called with invalid matrix
        # Then: A ValueError is raised
        with pytest.raises(ValueError, match="All inner text lists must have the same length"):
            rerank_by_keywords(
                texts=texts,
                seed_keywords=["fox"],
                top_n=2,
                show_progress=False
            )

    def test_empty_input_returns_empty_results(self, mock_dependencies):
        # Given: An empty list of texts
        mock_keybert, _, _ = mock_dependencies
        texts: List[str] = []

        # When: rerank_by_keywords is called with empty input
        results = rerank_by_keywords(
            texts=texts,
            seed_keywords=["fox"],
            top_n=2,
            show_progress=False
        )

        # Then: An empty result list is returned
        assert results == [], "Expected empty results for empty input"


class TestRerankByKeywordsSeedKeywords:
    def test_seed_keywords_filter_relevant_keywords(self, mock_dependencies):
        """Test that seed_keywords filter extracted keywords to only those containing seed terms."""
        # Given
        mock_keybert, _, _ = mock_dependencies
        texts = [
            "The quick brown fox jumps over the lazy dog",
            "A fast cat climbs steep hills",
            "The dog sleeps by the fire"
        ]
        seed_keywords = ["fox", "dog"]
        expected_keywords = [
            [("quick fox", 0.8), ("lazy dog", 0.7), ("brown fox", 0.6)],
            [("fast cat", 0.9), ("steep hills", 0.5)],
            [("dog sleeps", 0.6), ("fire place", 0.4)]
        ]
        expected_filtered_keywords = [
            [("quick fox", 0.8), ("lazy dog", 0.7)],
            [("dog sleeps", 0.6)],
            [],
        ]
        mock_keybert.extract_keywords.return_value = expected_keywords
        expected_result_length = len(texts)
        expected_scores = [0.8, 0.6, 0.0]

        # When
        results = rerank_by_keywords(
            texts=texts,
            seed_keywords=seed_keywords,
            top_n=2,
            show_progress=False
        )

        # Then
        assert len(
            results) == expected_result_length, f"Expected {expected_result_length} results, got {len(results)}"
        assert [r["score"]
                for r in results] == expected_scores, f"Expected scores {expected_scores}, got {[r['score'] for r in results]}"
        assert all(r["rank"] == i + 1 for i, r in enumerate(results)
                   ), "Ranks should be sequential starting from 1"
        assert all(r["text"] == texts[r["doc_index"]]
                   for r in results), "Text should match input text"
        assert [r["keywords"] for r in results] == [
            [{"text": kw, "score": score} for kw, score in kw_list]
            for kw_list in expected_filtered_keywords
        ], "Keywords should be filtered to include only those containing seed keywords"

    def test_empty_seed_keywords_returns_all_keywords(self, mock_dependencies):
        """Test that empty seed_keywords returns all extracted keywords without filtering."""
        # Given
        mock_keybert, _, _ = mock_dependencies
        texts = [
            "The quick brown fox jumps over the lazy dog",
            "A fast cat climbs steep hills"
        ]
        seed_keywords: List[str] = []
        expected_keywords = [
            [("fast cat", 0.9), ("steep hills", 0.5)],
            [("quick fox", 0.8), ("lazy dog", 0.7)],
        ]
        mock_keybert.extract_keywords.return_value = expected_keywords
        expected_result_length = len(texts)
        expected_scores = [0.9, 0.8]

        # When
        results = rerank_by_keywords(
            texts=texts,
            seed_keywords=seed_keywords,
            top_n=2,
            show_progress=False
        )

        # Then
        assert len(
            results) == expected_result_length, f"Expected {expected_result_length} results, got {len(results)}"
        assert [r["score"]
                for r in results] == expected_scores, f"Expected scores {expected_scores}, got {[r['score'] for r in results]}"
        assert all(r["rank"] == i + 1 for i, r in enumerate(results)
                   ), "Ranks should be sequential starting from 1"
        assert all(r["text"] == texts[r["doc_index"]]
                   for r in results), "Text should match input text"
        assert [r["keywords"] for r in results] == [
            [{"text": kw, "score": score} for kw, score in kw_list]
            for kw_list in expected_keywords
        ], "Keywords should include all extracted keywords when seed_keywords is empty"

    def test_matrix_seed_keywords_filter_relevant_keywords(self, mock_dependencies):
        """Test seed_keywords filtering with matrix input."""
        # Given
        mock_keybert, _, _ = mock_dependencies
        texts = [
            [
                "The quick brown fox jumps over the lazy dog",
                "Foxes are clever animals"
            ],
            [
                "A fast cat climbs steep hills",
                "The forest is home to quiet creatures"
            ]
        ]
        seed_keywords = ["fox", "dog"]
        expected_keywords = [
            [("quick fox", 0.8), ("lazy dog", 0.7), ("brown fox", 0.6)],
            [("clever foxes", 0.6), ("smart animals", 0.4)],
            [("fast cat", 0.9), ("steep hills", 0.5)],
            [("quiet creatures", 0.3), ("forest home", 0.2)]
        ]
        expected_filtered_keywords = [
            [("quick fox", 0.8), ("lazy dog", 0.7)],
            [("clever foxes", 0.6)],
            [],
            []
        ]
        mock_keybert.extract_keywords.return_value = expected_keywords
        expected_result_length = len(texts)
        expected_scores = [0.7, 0.0]

        # When
        results = rerank_by_keywords(
            texts=texts,
            seed_keywords=seed_keywords,
            top_n=2,
            show_progress=False
        )

        # Then
        assert len(
            results) == expected_result_length, f"Expected {expected_result_length} results, got {len(results)}"
        assert [round(r["score"], 4) for r in results] == [round(s, 4) for s in expected_scores], \
            f"Expected scores {[round(s, 4) for s in expected_scores]}, got {[round(r['score'], 4) for r in results]}"
        assert all(r["rank"] == i + 1 for i, r in enumerate(results)
                   ), "Ranks should be sequential starting from 1"
        assert all(r["text"] == " ".join(texts[r["doc_index"]])
                   for r in results), "Text should be joined input texts"
        assert [r["keywords"] for r in results] == [
            [
                {"text": kw, "score": score}
                for kw, score in sorted(
                    set(kw for sublist in expected_filtered_keywords[i*2:(
                        i+1)*2] for kw in sublist),
                    key=lambda x: x[1],
                    reverse=True
                )[:2]
            ]
            for i in range(len(texts))
        ], "Keywords should be filtered to include only those containing seed keywords"


class TestRerankByKeywordsCandidates:
    """Test suite for rerank_by_keywords with candidates parameter."""

    def test_candidates_restrict_keyword_extraction(self, mock_dependencies):
        """Test that candidates restrict extracted keywords to provided list."""
        # Given
        mock_keybert, _, _ = mock_dependencies
        texts = [
            "The quick brown fox jumps over the lazy dog",
            "A fast cat climbs steep hills",
            "The dog sleeps by the fire"
        ]
        candidates = ["quick fox", "lazy dog", "fast cat"]
        seed_keywords = ["fox", "dog"]
        expected_keywords = [
            [("quick fox", 0.8), ("lazy dog", 0.7)],
            [("fast cat", 0.9)],
            [("lazy dog", 0.6)]
        ]
        expected_result: List[SimilarityResult] = [
            {
                "id": mock.ANY,
                "rank": 1,
                "doc_index": 1,
                "score": 0.9,
                "text": "A fast cat climbs steep hills",
                "tokens": mock.ANY,
                "keywords": [{"text": "fast cat", "score": 0.9}]
            },
            {
                "id": mock.ANY,
                "rank": 2,
                "doc_index": 0,
                "score": 0.8,
                "text": "The quick brown fox jumps over the lazy dog",
                "tokens": mock.ANY,
                "keywords": [{"text": "quick fox", "score": 0.8}, {"text": "lazy dog", "score": 0.7}]
            },
            {
                "id": mock.ANY,
                "rank": 3,
                "doc_index": 2,
                "score": 0.6,
                "text": "The dog sleeps by the fire",
                "tokens": mock.ANY,
                "keywords": [{"text": "lazy dog", "score": 0.6}]
            }
        ]
        mock_keybert.extract_keywords.return_value = expected_keywords

        # When
        results = rerank_by_keywords(
            texts=texts,
            seed_keywords=seed_keywords,
            candidates=candidates,
            top_n=2,
            show_progress=False
        )

        # Then
        assert len(results) == len(
            texts), f"Expected {len(texts)} results, got {len(results)}"
        for i, result in enumerate(results):
            expected = expected_result[i]
            assert result["rank"] == expected[
                "rank"], f"Expected rank {expected['rank']} for result {i}, got {result['rank']}"
            assert result["doc_index"] == expected[
                "doc_index"], f"Expected doc_index {expected['doc_index']} for result {i}, got {result['doc_index']}"
            assert result["score"] == expected[
                "score"], f"Expected score {expected['score']} for result {i}, got {result['score']}"
            assert result["text"] == expected[
                "text"], f"Expected text {expected['text']} for result {i}, got {result['text']}"
            assert result["keywords"] == expected[
                "keywords"], f"Expected keywords {expected['keywords']} for result {i}, got {result['keywords']}"

    def test_candidates_with_min_count_filter(self, mock_dependencies):
        """Test that candidates are filtered by min_count parameter."""
        # Given
        mock_keybert, _, _ = mock_dependencies
        texts = [
            "The quick brown fox jumps over the lazy dog",
            "A fast cat climbs steep hills",
            "The dog sleeps by the fire"
        ]
        candidates = ["quick fox", "lazy dog", "rare term"]
        seed_keywords = ["fox", "dog"]
        # Simulate CountVectorizer output for min_count filtering
        mock_vectorizer = CountVectorizer()
        mock_vectorizer.get_feature_names_out.return_value = [
            "quick fox", "lazy dog"]  # "rare term" filtered out
        mock_vectorizer.fit_transform.return_value = np.array(
            [[1, 1, 0], [0, 0, 0], [0, 1, 0]])
        expected_keywords = [
            [("quick fox", 0.8), ("lazy dog", 0.7)],
            [],
            [("lazy dog", 0.6)]
        ]
        expected_result: List[SimilarityResult] = [
            {
                "id": mock.ANY,
                "rank": 1,
                "doc_index": 0,
                "score": 0.8,
                "text": "The quick brown fox jumps over the lazy dog",
                "tokens": mock.ANY,
                "keywords": [{"text": "quick fox", "score": 0.8}, {"text": "lazy dog", "score": 0.7}]
            },
            {
                "id": mock.ANY,
                "rank": 2,
                "doc_index": 2,
                "score": 0.6,
                "text": "The dog sleeps by the fire",
                "tokens": mock.ANY,
                "keywords": [{"text": "lazy dog", "score": 0.6}]
            },
            {
                "id": mock.ANY,
                "rank": 3,
                "doc_index": 1,
                "score": 0.0,
                "text": "A fast cat climbs steep hills",
                "tokens": mock.ANY,
                "keywords": []
            }
        ]
        mock_keybert.extract_keywords.return_value = expected_keywords

        # When
        results = rerank_by_keywords(
            texts=texts,
            seed_keywords=seed_keywords,
            candidates=candidates,
            min_count=2,  # "rare term" should be filtered out
            top_n=2,
            show_progress=False,
            vectorizer=mock_vectorizer
        )

        # Then
        assert len(results) == len(
            texts), f"Expected {len(texts)} results, got {len(results)}"
        for i, result in enumerate(results):
            expected = expected_result[i]
            assert result["rank"] == expected[
                "rank"], f"Expected rank {expected['rank']} for result {i}, got {result['rank']}"
            assert result["doc_index"] == expected[
                "doc_index"], f"Expected doc_index {expected['doc_index']} for result {i}, got {result['doc_index']}"
            assert result["score"] == expected[
                "score"], f"Expected score {expected['score']} for result {i}, got {result['score']}"
            assert result["text"] == expected[
                "text"], f"Expected text {expected['text']} for result {i}, got {result['text']}"
            assert result["keywords"] == expected[
                "keywords"], f"Expected keywords {expected['keywords']} for result {i}, got {result['keywords']}"
            assert all(kw["text"] in candidates[:2] for kw in result["keywords"]), \
                f"Keywords should only include valid candidates, got {result['keywords']}"

    def test_candidates_with_seed_keywords_and_threshold(self, mock_dependencies):
        """Test candidates with seed_keywords and threshold filtering."""
        # Given
        mock_keybert, _, _ = mock_dependencies
        texts = [
            "The quick brown fox jumps over the lazy dog",
            "A fast cat climbs steep hills"
        ]
        candidates = ["quick fox", "lazy dog", "fast cat", "steep hills"]
        seed_keywords = ["fox", "dog"]
        threshold = 0.7
        expected_keywords = [
            [("quick fox", 0.8), ("lazy dog", 0.7), ("brown fox", 0.6)],
            [("fast cat", 0.9), ("steep hills", 0.5)]
        ]
        # After threshold and seed_keywords filtering
        expected_filtered_keywords = [
            [("quick fox", 0.8), ("lazy dog", 0.7)],
            []
        ]
        expected_result: List[SimilarityResult] = [
            {
                "id": mock.ANY,
                "rank": 1,
                "doc_index": 0,
                "score": 0.8,
                "text": "The quick brown fox jumps over the lazy dog",
                "tokens": mock.ANY,
                "keywords": [{"text": "quick fox", "score": 0.8}, {"text": "lazy dog", "score": 0.7}]
            },
            {
                "id": mock.ANY,
                "rank": 2,
                "doc_index": 1,
                "score": 0.0,
                "text": "A fast cat climbs steep hills",
                "tokens": mock.ANY,
                "keywords": []
            }
        ]
        mock_keybert.extract_keywords.return_value = expected_keywords

        # When
        results = rerank_by_keywords(
            texts=texts,
            seed_keywords=seed_keywords,
            candidates=candidates,
            top_n=2,
            threshold=threshold,
            show_progress=False
        )

        # Then
        assert len(results) == len(
            texts), f"Expected {len(texts)} results, got {len(results)}"
        for i, result in enumerate(results):
            expected = expected_result[i]
            assert result["rank"] == expected[
                "rank"], f"Expected rank {expected['rank']} for result {i}, got {result['rank']}"
            assert result["doc_index"] == expected[
                "doc_index"], f"Expected doc_index {expected['doc_index']} for result {i}, got {result['doc_index']}"
            assert result["score"] == expected[
                "score"], f"Expected score {expected['score']} for result {i}, got {result['score']}"
            assert result["text"] == expected[
                "text"], f"Expected text {expected['text']} for result {i}, got {result['text']}"
            assert result["keywords"] == expected[
                "keywords"], f"Expected keywords {expected['keywords']} for result {i}, got {result['keywords']}"
            assert all(kw["score"] >= threshold for kw in result["keywords"]), \
                f"Keywords scores should be >= {threshold}, got {result['keywords']}"
            assert all(any(seed.lower() in kw["text"].lower() for seed in seed_keywords) for kw in result["keywords"]), \
                f"Keywords should contain seed keywords, got {result['keywords']}"
