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
        query = "teh"
        expected = [SearchResult(
            rank=1,
            score=pytest.approx(0.88, abs=1e-5),
            text="the",
            original="teh",
            source_document="teh",
            start_index=0,
            end_index=3
        )]
        results = vector_search.search(query, k=1)
        assert len(results) == 1
        assert results == expected

    def test_search_single_misspelled_word_no_k(self, vector_search):
        """Test searching a single misspelled word without k parameter."""
        query = "teh"
        expected = [SearchResult(
            rank=1,
            score=pytest.approx(0.88, abs=1e-5),
            text="the",
            original="teh",
            source_document="teh",
            start_index=0,
            end_index=3
        )]
        results = vector_search.search(query)
        assert len(results) == 1
        assert results == expected

    def test_search_correct_word(self, vector_search):
        """Test searching a correctly spelled word with custom dictionary."""
        query = "quick"
        expected = []
        results = vector_search.search(query, k=1)
        assert len(results) == 0
        assert results == expected

    def test_search_multiple_words_with_misspellings(self, vector_search):
        """Test searching a string with multiple words, some misspelled."""
        query = "teh quik foxx"
        expected = [
            SearchResult(
                rank=1,
                score=pytest.approx(0.88, abs=1e-5),
                text="the",
                original="teh",
                source_document="teh quik foxx",
                start_index=0,
                end_index=3
            ),
            SearchResult(
                rank=2,
                score=pytest.approx(0.24083853653640552, abs=1e-5),
                text="quick",
                original="quik",
                source_document="teh quik foxx",
                start_index=4,
                end_index=8
            ),
            SearchResult(
                rank=3,
                score=pytest.approx(0.24010421028462586, abs=1e-5),
                text="fox",
                original="foxx",
                source_document="teh quik foxx",
                start_index=9,
                end_index=13
            ),
        ]
        results = vector_search.search(query, k=3)
        assert len(results) <= 3
        assert results == expected

    def test_search_multiple_words_with_misspellings_no_k(self, vector_search):
        """Test searching a string with multiple words, some misspelled, without k parameter."""
        query = "teh quik foxx"
        expected = [
            SearchResult(
                rank=1,
                score=pytest.approx(0.88, abs=1e-5),
                text="the",
                original="teh",
                source_document="teh quik foxx",
                start_index=0,
                end_index=3
            ),
            SearchResult(
                rank=2,
                score=pytest.approx(0.24083853653640552, abs=1e-5),
                text="quick",
                original="quik",
                source_document="teh quik foxx",
                start_index=4,
                end_index=8
            ),
            SearchResult(
                rank=3,
                score=pytest.approx(0.24010421028462586, abs=1e-5),
                text="fox",
                original="foxx",
                source_document="teh quik foxx",
                start_index=9,
                end_index=13
            ),
        ]
        results = vector_search.search(query)
        assert len(results) == 3
        assert results == expected

    def test_search_list_of_documents(self, vector_search):
        """Test searching a list of documents with misspellings."""
        documents = ["teh quick foxx", "jummps over"]
        expected = [
            SearchResult(
                rank=1,
                score=pytest.approx(0.88, abs=1e-5),
                text="the",
                original="teh",
                source_document="teh quick foxx",
                start_index=0,
                end_index=3
            ),
            SearchResult(
                rank=2,
                score=pytest.approx(0.240104, abs=1e-5),
                text="fox",
                original="foxx",
                source_document="teh quick foxx",
                start_index=10,
                end_index=14
            ),
            SearchResult(
                rank=3,
                score=pytest.approx(0.240091, abs=1e-5),
                text="jumps",
                original="jummps",
                source_document="jummps over",
                start_index=0,
                end_index=6
            ),
        ]
        results = vector_search.search(documents, k=4)
        assert len(results) <= 4
        assert results == expected

    def test_search_list_of_documents_no_k(self, vector_search):
        """Test searching a list of documents with misspellings without k parameter."""
        documents = ["teh quick foxx", "jummps over"]
        expected = [
            SearchResult(
                rank=1,
                score=pytest.approx(0.88, abs=1e-5),
                text="the",
                original="teh",
                source_document="teh quick foxx",
                start_index=0,
                end_index=3
            ),
            SearchResult(
                rank=2,
                score=pytest.approx(0.240104, abs=1e-5),
                text="fox",
                original="foxx",
                source_document="teh quick foxx",
                start_index=10,
                end_index=14
            ),
            SearchResult(
                rank=3,
                score=pytest.approx(0.240091, abs=1e-5),
                text="jumps",
                original="jummps",
                source_document="jummps over",
                start_index=0,
                end_index=6
            ),
        ]
        results = vector_search.search(documents)
        assert len(results) == 3
        assert results == expected

    def test_search_with_default_dictionary(self, vector_search_default):
        """Test searching with default dictionary."""
        query = "teh"
        expected = [SearchResult(
            rank=1,
            score=pytest.approx(0.88, abs=1e-5),
            text="the",
            original="teh",
            source_document="teh",
            start_index=0,
            end_index=3
        )]
        results = vector_search_default.search(query, k=1)
        assert len(results) == 1
        assert results == expected

    @pytest.mark.parametrize("invalid_input", ["", [], [""]])
    def test_search_with_invalid_input(self, vector_search, invalid_input):
        """Test searching with empty or invalid input."""
        query = invalid_input
        expected = []
        results = vector_search.search(query, k=1)
        assert len(results) == 0
        assert results == expected
