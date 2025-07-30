import pytest
from typing import List
from jet.wordnet.keywords.keyword_search_spellchecker import KeywordVectorSearchSpellChecker, SearchResult


@pytest.fixture
def vector_search():
    words = ["the", "quick", "brown", "fox", "jumps"]
    searcher = KeywordVectorSearchSpellChecker()
    searcher.build_index(words)
    return searcher


@pytest.fixture
def vector_search_default():
    searcher = KeywordVectorSearchSpellChecker()
    searcher.build_index()
    return searcher


def test_search_with_misspelled_word(vector_search):
    # Given a misspelled query word
    query = "teh"
    expected = [SearchResult(
        rank=1, score=pytest.approx(0.0, abs=1.0), text="the")]

    # When searching for the query
    results = vector_search.search(query, k=1)

    # Then the result should match the expected word with a valid score
    assert results[0]["text"] == expected[0]["text"]
    assert results[0]["score"] >= 0.0
    assert results[0]["rank"] == expected[0]["rank"]


def test_search_with_correct_word(vector_search):
    # Given a correctly spelled query word
    query = "quick"
    expected = [SearchResult(
        rank=1, score=pytest.approx(0.0, abs=1.0), text="quick")]

    # When searching for the query
    results = vector_search.search(query, k=1)

    # Then the result should match the expected word with a valid score
    assert results[0]["text"] == expected[0]["text"]
    assert results[0]["score"] > 0.0
    assert results[0]["rank"] == expected[0]["rank"]


def test_search_without_build_index():
    # Given a searcher without explicitly calling build_index
    searcher = KeywordVectorSearchSpellChecker()
    query = "teh"
    expected = [SearchResult(
        rank=1, score=pytest.approx(0.0, abs=1.0), text="the")]

    # When searching for a misspelled word
    results = searcher.search(query, k=1)

    # Then the result should match the expected word with a valid score
    assert results[0]["text"] == expected[0]["text"]
    assert results[0]["score"] >= 0.0
    assert results[0]["rank"] == expected[0]["rank"]


def test_search_results_sorted_by_score(vector_search):
    # Given a query with multiple results
    query = "quik"
    expected = [
        SearchResult(rank=1, score=pytest.approx(0.0, abs=1.0), text="quick"),
        SearchResult(rank=2, score=pytest.approx(0.0, abs=1.0), text="brown"),
        SearchResult(rank=3, score=pytest.approx(0.0, abs=1.0), text="fox")
    ]

    # When searching for the query with multiple results
    results = vector_search.search(query, k=3)

    # Then results should be sorted by score in descending order
    assert len(results) <= 3
    assert results[0]["text"] == expected[0]["text"]
    assert results[0]["score"] >= results[1]["score"] if len(
        results) > 1 else True
    assert results[1]["score"] >= results[2]["score"] if len(
        results) > 2 else True
    assert results[0]["rank"] == 1
    assert results[1]["rank"] == 2 if len(results) > 1 else True
    assert results[2]["rank"] == 3 if len(results) > 2 else True


def test_search_with_default_dictionary(vector_search_default):
    # Given a searcher with the default English dictionary
    query = "teh"
    expected = [SearchResult(
        rank=1, score=pytest.approx(0.0, abs=1.0), text="the")]

    # When searching for a misspelled word
    results = vector_search_default.search(query, k=1)

    # Then the result should match the expected word with a valid score
    assert results[0]["text"] == expected[0]["text"]
    assert results[0]["score"] >= 0.0
    assert results[0]["rank"] == expected[0]["rank"]
