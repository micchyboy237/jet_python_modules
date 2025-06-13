import pytest
from jet.logger import logger
from jet.vectors.document_types import HeaderDocument
from jet.wordnet.text_chunker import chunk_headers, chunk_sentences_with_indices, chunk_texts, chunk_sentences, chunk_texts_with_indices, truncate_texts


class TestChunkTexts:
    def test_no_overlap(self):
        input_text = "This is a test text with several words to be chunked into smaller pieces"
        expected = [
            "This is a test text with several words",
            "to be chunked into smaller pieces"
        ]
        result = chunk_texts(input_text, chunk_size=8, chunk_overlap=0)
        assert result == expected

    def test_with_overlap(self):
        input_text = "This is a test text with several words to be chunked"
        expected = [
            "This is a test text with several words",
            "with several words to be chunked"
        ]
        result = chunk_texts(input_text, chunk_size=8, chunk_overlap=3)
        assert result == expected


class TestChunkSentences:
    def test_no_overlap(self):
        input_text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
        expected = [
            "This is sentence one. This is sentence two.",
            "This is sentence three. This is sentence four."
        ]
        result = chunk_sentences(input_text, chunk_size=2, sentence_overlap=0)
        assert result == expected

    def test_with_overlap(self):
        input_text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
        expected = [
            "This is sentence one. This is sentence two.",
            "This is sentence two. This is sentence three.",
            "This is sentence three. This is sentence four."
        ]
        result = chunk_sentences(input_text, chunk_size=2, sentence_overlap=1)
        assert result == expected


class TestChunkTextsWithIndices:
    def test_no_overlap(self):
        input_text = "This is a test text with several words to be chunked into smaller pieces"
        expected = [
            "This is a test text with several words",
            "to be chunked into smaller pieces"
        ]
        expected_indices = [0, 0]
        result, result_indices = chunk_texts_with_indices(
            input_text, chunk_size=8, chunk_overlap=0)
        assert result == expected
        assert result_indices == expected_indices

    def test_with_overlap(self):
        input_text = "This is a test text with several words to be chunked"
        expected = [
            "This is a test text with several words",
            "with several words to be chunked"
        ]
        expected_indices = [0, 0]
        result, result_indices = chunk_texts_with_indices(
            input_text, chunk_size=8, chunk_overlap=3)
        assert result == expected
        assert result_indices == expected_indices


class TestChunkSentencesWithIndices:
    def test_no_overlap(self):
        input_text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
        expected = [
            "This is sentence one. This is sentence two.",
            "This is sentence three. This is sentence four."
        ]
        expected_indices = [0, 0]
        result, result_indices = chunk_sentences_with_indices(
            input_text, chunk_size=2, sentence_overlap=0)
        assert result == expected
        assert result_indices == expected_indices

    def test_with_overlap(self):
        input_text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
        expected = [
            "This is sentence one. This is sentence two.",
            "This is sentence two. This is sentence three.",
            "This is sentence three. This is sentence four."
        ]
        expected_indices = [0, 0, 0]
        result, result_indices = chunk_sentences_with_indices(
            input_text, chunk_size=2, sentence_overlap=1)
        assert result == expected
        assert result_indices == expected_indices


class TestChunkHeaders:
    def test_chunk_headers_single_doc(self):
        doc = HeaderDocument(
            id="doc1",
            text="Line 1\nLine 2\nLine 3\nLine 4",
            metadata={"doc_index": 1, "parent_header": "Parent"}
        )
        # Small max_tokens for testing
        result = chunk_headers([doc], max_tokens=2)
        expected = [
            HeaderDocument(
                id="doc1_chunk_0",
                text="Line 1\nLine 2",
                metadata={
                    "header": "Line 1...",
                    "parent_header": "Parent",
                    "header_level": 1,
                    "content": "Line 1\nLine 2",
                    "doc_index": 1,
                    "chunk_index": 0,
                    "texts": ["Line 1", "Line 2"]
                }
            ),
            HeaderDocument(
                id="doc1_chunk_1",
                text="Line 3\nLine 4",
                metadata={
                    "header": "Line 3...",
                    "parent_header": "Parent",
                    "header_level": 1,
                    "content": "Line 3\nLine 4",
                    "doc_index": 1,
                    "chunk_index": 1,
                    "texts": ["Line 3", "Line 4"]
                }
            )
        ]
        for r, e in zip(result, expected):
            assert r.id == e.id
            assert r.text == e.text
            assert r.metadata == e.metadata
        logger.debug("Test chunk_headers_single_doc passed")


class TestTruncateText:
    def test_single_string(self):
        sample = "This is the first sentence. Here is the second one. This is the third and final sentence."
        expected = "This is the first sentence. Here is the second one."
        result = truncate_texts(sample, max_words=12)
        assert result == expected

    def test_exact_word_limit(self):
        sample = "One. Two three four five. Six seven eight nine ten."
        expected = "One. Two three four five."
        result = truncate_texts(sample, max_words=6)
        assert result == expected

    def test_list_input(self):
        sample = [
            "First sentence. Second sentence. Third sentence.",
            "Another paragraph. With more text."
        ]
        expected = [
            "First sentence. Second sentence.",
            "Another paragraph."
        ]
        result = truncate_texts(sample, max_words=4)
        assert result == expected

    def test_short_text(self):
        sample = "Short text."
        expected = "Short text."
        result = truncate_texts(sample, max_words=100)
        assert result == expected
