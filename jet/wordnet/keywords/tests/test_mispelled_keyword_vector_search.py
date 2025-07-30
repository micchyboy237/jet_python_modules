import pytest
from typing import List
from jet.wordnet.keywords.mispelled_keyword_vector_search import MispelledKeywordVectorSearch, SearchResult


@pytest.fixture
def vector_search():
    """Fixture for a vector search with a custom word list."""
    words = ["the", "quick", "brown", "fox", "jumps"]
    searcher = MispelledKeywordVectorSearch()
    searcher.build_index(words)
    return searcher


@pytest.fixture
def vector_search_default():
    """Fixture for a vector search with default dictionary."""
    searcher = MispelledKeywordVectorSearch()
    searcher.build_index()
    return searcher


class TestMispelledKeywordVectorSearch:
    def test_search_single_misspelled_word(self, vector_search):
        """Test searching a single misspelled word with custom dictionary."""
        # Given
        query = "teh"
        expected = [SearchResult(rank=1, score=pytest.approx(
            0.0, abs=1.0), text="the", original="teh")]

        # When
        results = vector_search.search(query, k=1)

        # Then
        assert len(results) == 1
        assert results == expected

    def test_search_correct_word(self, vector_search):
        """Test searching a correctly spelled word with custom dictionary."""
        # Given
        query = "quick"
        expected = []  # No results for correct words

        # When
        results = vector_search.search(query, k=1)

        # Then
        assert len(results) == 0
        assert results == expected

    def test_search_multiple_words_with_misspellings(self, vector_search):
        """Test searching a string with multiple words, some misspelled."""
        # Given
        query = "teh quik foxx"
        expected = [
            SearchResult(rank=1, score=pytest.approx(
                0.0, abs=1.0), text="the", original="teh"),
            SearchResult(rank=2, score=pytest.approx(
                0.0, abs=1.0), text="quick", original="quik"),
            SearchResult(rank=3, score=pytest.approx(
                0.0, abs=1.0), text="fox", original="foxx"),
        ]

        # When
        results = vector_search.search(query, k=3)

        # Then
        assert len(results) <= 3
        assert results == expected

    def test_search_list_of_documents(self, vector_search):
        """Test searching a list of documents with misspellings."""
        # Given
        documents = ["teh quick foxx", "jummps over"]
        expected = [
            SearchResult(rank=1, score=pytest.approx(
                0.046241, abs=1e-5), text="the", original="teh"),
            SearchResult(rank=2, score=pytest.approx(
                6.883989e-06, abs=1e-5), text="fox", original="foxx"),
            SearchResult(rank=3, score=pytest.approx(
                6.029487e-06, abs=1e-5), text="jumps", original="jummps"),
        ]

        # When
        results = vector_search.search(documents, k=4)

        # Then
        assert len(results) <= 4
        assert results == expected

    def test_search_with_default_dictionary(self, vector_search_default):
        """Test searching with default dictionary."""
        # Given
        query = "teh"
        expected = [SearchResult(rank=1, score=pytest.approx(
            0.0, abs=1.0), text="the", original="teh")]

        # When
        results = vector_search_default.search(query, k=1)

        # Then
        assert len(results) == 1
        assert results == expected

    @pytest.mark.parametrize("invalid_input", ["", [], [""]])
    def test_search_with_invalid_input(self, vector_search, invalid_input):
        """Test searching with empty or invalid input."""
        # Given
        query = invalid_input
        expected = []

        # When
        results = vector_search.search(query, k=1)

        # Then
        assert len(results) == 0
        assert results == expected
