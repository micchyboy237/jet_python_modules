# jet/llm/rag/tests/test_rag_preprocessor.py
import pytest
from jet.llm.rag.preprocessors.web import WebDataPreprocessor
from typing import List


@pytest.fixture
def preprocessor():
    """Fixture to initialize WebDataPreprocessor."""
    return WebDataPreprocessor(chunk_size=500, chunk_overlap=50)


class TestWebDataPreprocessor:
    """Test suite for WebDataPreprocessor using BDD principles."""

    def test_fetch_and_extract_valid_url(self, preprocessor: WebDataPreprocessor):
        """Test fetching and extracting content from a valid URL."""
        # Given a valid URL
        url = "https://example.com"
        # When fetching and extracting content
        result = preprocessor.fetch_and_extract(url)
        # Then content should be extracted successfully
        expected = "This domain is for use"
        assert result is not None, "Content should not be None"
        assert expected in result, f"Expected '{expected}' in extracted content"

    def test_fetch_and_extract_invalid_url(self, preprocessor: WebDataPreprocessor):
        """Test fetching and extracting content from an invalid URL."""
        # Given an invalid URL
        url = "https://nonexistent.example.com"
        # When fetching and extracting content
        result = preprocessor.fetch_and_extract(url)
        # Then result should be None
        expected = None
        assert result == expected, f"Expected {expected}, got {result}"

    def test_clean_text(self, preprocessor: WebDataPreprocessor):
        """Test text cleaning functionality."""
        # Given a sample text with noise
        sample_text = """This is a  test! With URLs: https://example.com, emails: test@example.com,
                        and extra   spaces."""
        # When cleaning the text
        result = preprocessor.clean_text(sample_text)
        # Then text should be cleaned appropriately
        expected = "This is a test With URLs emails and extra spaces"
        assert result == expected, f"Expected '{expected}', got '{result}'"

    def test_chunk_text(self, preprocessor: WebDataPreprocessor):
        """Test text chunking functionality."""
        # Given a sample text
        sample_text = "This is sentence one. This is sentence two. This is a longer sentence to test chunking."
        # When chunking the text
        result: List[str] = preprocessor.chunk_text(sample_text)
        # Then chunks should be generated correctly
        expected = [
            "This is sentence one. This is sentence two. This is a longer sentence to test chunking."]
        assert result == expected, f"Expected {expected}, got {result}"
