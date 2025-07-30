import pytest
from typing import List
from jet.wordnet.keywords.keyword_search_t5 import KeywordVectorSearch, SearchResult


@pytest.fixture
def vector_search():
    words = ["the", "quick", "brown", "fox", "jumps"]
    searcher = KeywordVectorSearch()
    searcher.build_index(words)
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
        rank=1, score=pytest.approx(0.0, abs=1e-5), text="quick")]

    # When searching for the query
    results = vector_search.search(query, k=1)

    # Then the result should match the expected word with a near-zero score
    assert results[0]["text"] == expected[0]["text"]
    assert results[0]["score"] < 1e-5
    assert results[0]["rank"] == expected[0]["rank"]


def test_search_without_index_raises_error():
    # Given an uninitialized searcher
    searcher = KeywordVectorSearch()
    query = "test"

    # When searching without building an index
    with pytest.raises(ValueError) as exc_info:
        searcher.search(query, k=1)

    # Then it should raise the expected error
    expected_error = "Index or word list not initialized. Call build_index first."
    assert str(exc_info.value) == expected_error


def test_search_results_sorted_by_score(vector_search):
    # Given a query with multiple results
    query = "quick"
    expected = [
        SearchResult(rank=1, score=pytest.approx(0.0, abs=1e-5), text="quick"),
        SearchResult(rank=2, score=pytest.approx(1.0, abs=1.0), text="brown"),
        SearchResult(rank=3, score=pytest.approx(1.0, abs=1.0), text="fox")
    ]

    # When searching for the query with multiple results
    results = vector_search.search(query, k=3)

    # Then results should be sorted by score in descending order
    assert len(results) == 3
    assert results[0]["text"] == expected[0]["text"]
    assert results[0]["score"] <= results[1]["score"] <= results[2]["score"]
    assert results[0]["rank"] == 1
    assert results[1]["rank"] == 2
    assert results[2]["rank"] == 3
