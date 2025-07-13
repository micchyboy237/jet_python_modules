from jet.wordnet.keywords.helpers import extract_keyword_candidates
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from typing import List
import pytest


class TestExtractKeywordCandidates:
    def test_extracts_keywords_from_multiple_texts(self):
        # Given: A list of texts with common terms representing machine learning topics
        texts = [
            "machine learning is fun",
            "learning algorithms are key",
            "machine algorithms improve performance"
        ]
        expected = ["machine learning", "learning algorithms",
                    "machine", "learning", "algorithms"]
        # When: Extracting keywords with default settings
        result = extract_keyword_candidates(
            texts, ngram_range=(1, 2), stop_words="english", min_df=1)
        # Then: Should return expected keywords, prioritizing bigrams
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_top_n_keywords(self):
        # Given: A list of texts and top_n=2
        texts = [
            "machine learning is fun",
            "learning algorithms are key",
            "machine algorithms improve performance"
        ]
        expected = ["machine learning", "learning algorithms"]
        # When: Extracting top 2 keywords
        result = extract_keyword_candidates(texts, ngram_range=(
            1, 2), stop_words="english", min_df=1, top_n=2)
        # Then: Should return top 2 bigrams as they are more representative
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_handles_insufficient_documents(self):
        # Given: A single text, min_df=2 (more than available documents)
        texts = ["machine learning is fun"]
        expected = ["machine learning", "machine", "learning", "fun"]
        # When: Extracting keywords with min_df adjusted automatically
        result = extract_keyword_candidates(
            texts, ngram_range=(1, 2), stop_words="english", min_df=2)
        # Then: Should return all keywords since min_df is adjusted to 1
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_handles_empty_input(self):
        # Given: An empty list of texts
        texts: List[str] = []
        expected: List[str] = []
        # When: Extracting keywords from empty input
        result = extract_keyword_candidates(
            texts, ngram_range=(1, 2), stop_words="english", min_df=2)
        # Then: Should return empty list
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_handles_invalid_texts(self):
        # Given: A list with invalid (empty) texts
        texts = ["", ""]
        expected: List[str] = []
        # When: Extracting keywords from invalid texts
        result = extract_keyword_candidates(
            texts, ngram_range=(1, 2), stop_words="english", min_df=1)
        # Then: Should return empty list due to vectorization failure
        assert result == expected, f"Expected {expected}, but got {result}"
