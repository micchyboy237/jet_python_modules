import pytest
import nltk
from typing import List, TypedDict, Optional
from jet.vectors.semantic_search.vector_search_web import chunk_with_overlap, SentenceTransformer
import uuid
import re


@pytest.fixture(scope="module")
def setup_nltk():
    nltk.download('punkt', quiet=True)


@pytest.fixture
def model():
    return SentenceTransformer('all-MiniLM-L6-v2')


class Chunk(TypedDict):
    chunk_id: str
    text: str
    token_count: int
    header: Optional[str]
    header_level: Optional[str]
    xpath: Optional[str]


class TestChunkWithOverlap:
    def test_chunk_with_overlap_typical_case(self, setup_nltk, model):
        # Given: A section with a header and content within max_tokens
        section = {
            "header": "Python Basics",
            "header_level": "h2",
            "content": [
                "Python is a high-level programming language.",
                "It is used for web development and data science."
            ],
            "xpath": "//h2"
        }
        max_tokens = 200
        overlap_tokens = 50
        expected = [
            {
                "chunk_id": None,  # Will be checked separately
                "text": "Python is a high-level programming language. It is used for web development and data science.",
                "token_count": 17,
                "header": "Python Basics",
                "header_level": "h2",
                "xpath": "//h2"
            }
        ]

        # When: Chunking the section
        results: List[Chunk] = chunk_with_overlap(
            section, max_tokens, overlap_tokens, model)

        # Then: Expect one chunk with combined content
        assert len(results) == 1, "Expected exactly one chunk"
        result = results[0]
        assert isinstance(result["chunk_id"],
                          str), "Chunk ID should be a string"
        assert re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
                        result["chunk_id"]), "Chunk ID should be a valid UUID"
        assert result["text"] == expected[0]["text"], f"Expected text {expected[0]['text']}"
        assert result["token_count"] == expected[0][
            "token_count"], f"Expected token_count {expected[0]['token_count']}"
        assert result["header"] == expected[0][
            "header"], f"Expected header {expected[0]['header']}"
        assert result["header_level"] == expected[0][
            "header_level"], f"Expected header_level {expected[0]['header_level']}"
        assert result["xpath"] == expected[0][
            "xpath"], f"Expected xpath {expected[0]['xpath']}"

    def test_chunk_with_overlap_exceeds_max_tokens(self, setup_nltk, model):
        # Given: A section with content exceeding max_tokens, requiring overlap
        section = {
            "header": "Python Overview",
            "header_level": "h1",
            "content": [
                "Python is a high-level, interpreted programming language known for its simplicity and readability.",
                "It supports multiple programming paradigms, including object-oriented, functional, and procedural styles.",
                "Python is widely used in web development, data science, automation, and machine learning applications."
            ],
            "xpath": "//h1"
        }
        max_tokens = 20
        overlap_tokens = 5
        expected = [
            {
                "chunk_id": None,  # Will be checked separately
                "text": "Python is a high-level, interpreted programming language known for its simplicity and readability.",
                "token_count": 15,
                "header": "Python Overview",
                "header_level": "h1",
                "xpath": "//h1"
            },
            {
                "chunk_id": None,
                "text": "readability. It supports multiple programming paradigms, including object-oriented, functional, and procedural styles.",
                "token_count": 17,
                "header": "Python Overview",
                "header_level": "h1",
                "xpath": "//h1"
            },
            {
                "chunk_id": None,
                "text": "procedural styles. Python is widely used in web development, data science, automation, and machine learning applications.",
                "token_count": 18,
                "header": "Python Overview",
                "header_level": "h1",
                "xpath": "//h1"
            }
        ]

        # When: Chunking the section
        results: List[Chunk] = chunk_with_overlap(
            section, max_tokens, overlap_tokens, model)

        # Then: Expect three chunks with overlap
        assert len(results) == 3, "Expected three chunks due to token limit"
        for i, result in enumerate(results):
            assert isinstance(result["chunk_id"],
                              str), f"Chunk ID {i} should be a string"
            assert re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
                            result["chunk_id"]), f"Chunk ID {i} should be a valid UUID"
            assert result["text"] == expected[i][
                "text"], f"Expected text {expected[i]['text']} for chunk {i}"
            assert result["token_count"] == expected[i][
                "token_count"], f"Expected token_count {expected[i]['token_count']} for chunk {i}"
            assert result["header"] == expected[i][
                "header"], f"Expected header {expected[i]['header']} for chunk {i}"
            assert result["header_level"] == expected[i][
                "header_level"], f"Expected header_level {expected[i]['header_level']} for chunk {i}"
            assert result["xpath"] == expected[i][
                "xpath"], f"Expected xpath {expected[i]['xpath']} for chunk {i}"

    def test_chunk_with_overlap_no_header(self, setup_nltk, model):
        # Given: A section with no header and short content
        section = {
            "header": None,
            "header_level": None,
            "content": ["Python is a versatile language."],
            "xpath": None
        }
        max_tokens = 200
        overlap_tokens = 50
        expected = [
            {
                "chunk_id": None,  # Will be checked separately
                "text": "Python is a versatile language.",
                "token_count": 6,
                "header": None,
                "header_level": None,
                "xpath": None
            }
        ]

        # When: Chunking the section
        results: List[Chunk] = chunk_with_overlap(
            section, max_tokens, overlap_tokens, model)

        # Then: Expect one chunk without header
        assert len(results) == 1, "Expected exactly one chunk"
        result = results[0]
        assert isinstance(result["chunk_id"],
                          str), "Chunk ID should be a string"
        assert re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
                        result["chunk_id"]), "Chunk ID should be a valid UUID"
        assert result["text"] == expected[0]["text"], f"Expected text {expected[0]['text']}"
        assert result["token_count"] == expected[0][
            "token_count"], f"Expected token_count {expected[0]['token_count']}"
        assert result["header"] is None, "Expected no header"
        assert result["header_level"] is None, "Expected no header_level"
        assert result["xpath"] is None, "Expected no xpath"

    def test_chunk_with_overlap_short_content(self, setup_nltk, model):
        # Given: A section with content too short to chunk
        section = {
            "header": "Short Section",
            "header_level": "h3",
            "content": ["Short."],
            "xpath": "//h3"
        }
        max_tokens = 200
        overlap_tokens = 50
        expected: List[Chunk] = []

        # When: Chunking the section
        results: List[Chunk] = chunk_with_overlap(
            section, max_tokens, overlap_tokens, model)

        # Then: Expect no chunks due to minimum length (5 tokens)
        assert results == expected, "Expected no chunks due to short content"

    def test_chunk_with_overlap_low_similarity(self, setup_nltk, model):
        # Given: A section with content unrelated to the header
        section = {
            "header": "Python Programming",
            "header_level": "h2",
            "content": ["This is about Java programming, not Python."],
            "xpath": "//h2"
        }
        max_tokens = 200
        overlap_tokens = 50
        expected: List[Chunk] = []

        # When: Chunking the section
        results: List[Chunk] = chunk_with_overlap(
            section, max_tokens, overlap_tokens, model)

        # Then: Expect no chunks due to low similarity
        assert results == expected, "Expected no chunks due to low header-content similarity"

    def test_chunk_with_overlap_empty_content(self, setup_nltk, model):
        # Given: A section with a header but no content
        section = {
            "header": "Python Introduction",
            "header_level": "h1",
            "content": [],
            "xpath": "//h1"
        }
        max_tokens = 200
        overlap_tokens = 50
        expected: List[Chunk] = []

        # When: Chunking the section
        results: List[Chunk] = chunk_with_overlap(
            section, max_tokens, overlap_tokens, model)

        # Then: Expect no chunks due to empty content
        assert results == expected, "Expected no chunks due to empty content"
