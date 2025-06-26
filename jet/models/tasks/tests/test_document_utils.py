from typing import List, Dict, Any
import pytest
from jet.wordnet.words import get_words
from jet.models.tasks.hybrid_search_docs_with_bm25 import split_document


class TestSplitDocument:
    def test_splits_document_with_headers_and_content(self):
        # Given: A document with headers and content
        doc_text = "# Header 1\nFirst sentence. Second sentence.\n## Header 2\nThird sentence. Fourth sentence."
        doc_id = "doc_1"
        doc_index = 0
        chunk_size = 10  # Small chunk size to force splitting
        overlap = 0
        expected = [
            {
                "text": "# Header 1\nFirst sentence. Second sentence.",
                "headers": ["# Header 1"],
                "doc_id": "doc_1",
                "doc_index": 0
            },
            {
                "text": "## Header 2\nThird sentence. Fourth sentence.",
                "headers": ["## Header 2"],
                "doc_id": "doc_1",
                "doc_index": 0
            }
        ]

        # When: Splitting the document
        result = split_document(
            doc_text, doc_id, doc_index, chunk_size, overlap)

        # Then: The document is split correctly with headers preserved
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_handles_empty_document(self):
        # Given: An empty document
        doc_text = ""
        doc_id = "doc_2"
        doc_index = 1
        chunk_size = 800
        overlap = 200
        expected: List[Dict[str, Any]] = []

        # When: Splitting the document
        result = split_document(
            doc_text, doc_id, doc_index, chunk_size, overlap)

        # Then: No chunks are returned
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_splits_with_overlap(self):
        # Given: A document with content that requires overlap
        doc_text = "First sentence. Second sentence. Third sentence."
        doc_id = "doc_3"
        doc_index = 2
        chunk_size = 5  # Small chunk size to force splitting
        overlap = 2
        expected = [
            {
                "text": "First sentence. Second sentence.",
                "headers": [],
                "doc_id": "doc_3",
                "doc_index": 2
            },
            {
                "text": "Second sentence. Third sentence.",
                "headers": [],
                "doc_id": "doc_3",
                "doc_index": 2
            }
        ]

        # When: Splitting the document with overlap
        result = split_document(
            doc_text, doc_id, doc_index, chunk_size, overlap)

        # Then: Chunks include overlap correctly
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_handles_single_header_no_content(self):
        # Given: A document with only a header
        doc_text = "# Header Only"
        doc_id = "doc_4"
        doc_index = 3
        chunk_size = 800
        overlap = 200
        expected = [
            {
                "text": "# Header Only",
                "headers": ["# Header Only"],
                "doc_id": "doc_4",
                "doc_index": 3
            }
        ]

        # When: Splitting the document
        result = split_document(
            doc_text, doc_id, doc_index, chunk_size, overlap)

        # Then: The header is returned as a single chunk
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_handles_single_sentence(self):
        # Given: A document with a single sentence
        doc_text = "Single sentence."
        doc_id = "doc_5"
        doc_index = 4
        chunk_size = 800
        overlap = 0
        expected = [
            {
                "text": "Single sentence.",
                "headers": [],
                "doc_id": "doc_5",
                "doc_index": 4
            }
        ]

        # When: Splitting the document
        result = split_document(
            doc_text, doc_id, doc_index, chunk_size, overlap)

        # Then: The single sentence is returned as a single chunk
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_handles_multiple_sentences_in_chunk(self):
        # Given: A document with multiple sentences fitting within chunk size
        doc_text = "First sentence. Second sentence. Third sentence."
        doc_id = "doc_6"
        doc_index = 5
        chunk_size = 50  # Large enough to fit all sentences
        overlap = 0
        expected = [
            {
                "text": "First sentence. Second sentence. Third sentence.",
                "headers": [],
                "doc_id": "doc_6",
                "doc_index": 5
            }
        ]

        # When: Splitting the document
        result = split_document(
            doc_text, doc_id, doc_index, chunk_size, overlap)

        # Then: All sentences are in a single chunk
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_ignores_empty_lines(self):
        # Given: A document with empty lines
        doc_text = "# Header\n\nFirst sentence.\n\nSecond sentence."
        doc_id = "doc_7"
        doc_index = 6
        chunk_size = 800
        overlap = 0
        expected = [
            {
                "text": "# Header\nFirst sentence. Second sentence.",
                "headers": ["# Header"],
                "doc_id": "doc_7",
                "doc_index": 6
            }
        ]

        # When: Splitting the document
        result = split_document(
            doc_text, doc_id, doc_index, chunk_size, overlap)

        # Then: Empty lines are ignored, and content is combined
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_splits_large_document(self):
        # Given: A large document exceeding chunk size
        doc_text = "# Header\n" + " ".join(["word"] * 1000)  # ~1000 words
        doc_id = "doc_8"
        doc_index = 7
        chunk_size = 200
        overlap = 50
        result = split_document(
            doc_text, doc_id, doc_index, chunk_size, overlap)

        # When: Splitting the document
        expected_first_chunk = {
            "text": "# Header\n" + " ".join(["word"] * (200 - len(get_words("# Header")))),
            "headers": ["# Header"],
            "doc_id": "doc_8",
            "doc_index": 7
        }
        expected_second_chunk = {
            "text": " ".join(["word"] * 50) + " " + " ".join(["word"] * (200 - 50)),
            "headers": ["# Header"],
            "doc_id": "doc_8",
            "doc_index": 7
        }

        # Then: Document is split into multiple chunks with overlap
        assert len(result) > 1, "Expected multiple chunks"
        assert result[0] == expected_first_chunk, f"First chunk mismatch: {result[0]}"
        assert result[1] == expected_second_chunk, f"Second chunk mismatch: {result[1]}"
        assert result[1]["text"].startswith(
            " ".join(["word"] * 50)), "Overlap not applied correctly"
        assert result[1]["headers"] == [
            "# Header"], "Headers not preserved in second chunk"
