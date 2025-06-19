import pytest
from jet.logger import logger
from jet.vectors.document_types import HeaderDocument
from jet.wordnet.sentence import split_sentences
from jet.wordnet.text_chunker import chunk_headers, chunk_sentences_with_indices, chunk_texts, chunk_sentences, chunk_texts_with_indices, truncate_texts
from jet.models.tokenizer.base import detokenize, get_tokenizer_fn

MODEL_NAME = "qwen3-1.7b-4bit"


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

    def test_no_overlap_with_model(self):
        input_text = "This is a test text with several words to be chunked into smaller pieces"
        tokenize_fn = get_tokenizer_fn(MODEL_NAME)
        token_ids = tokenize_fn(input_text)
        expected_chunks = [
            detokenize(token_ids[:8], MODEL_NAME),
            detokenize(token_ids[8:], MODEL_NAME)
        ]
        result = chunk_texts(input_text, chunk_size=8,
                             chunk_overlap=0, model=MODEL_NAME)
        assert result == expected_chunks

    def test_with_overlap_with_model(self):
        input_text = "This is a test text with several words to be chunked"
        tokenize_fn = get_tokenizer_fn(MODEL_NAME)
        token_ids = tokenize_fn(input_text)
        expected_chunks = [
            detokenize(token_ids[:8], MODEL_NAME),
            detokenize(token_ids[5:], MODEL_NAME)
        ]
        result = chunk_texts(input_text, chunk_size=8,
                             chunk_overlap=3, model=MODEL_NAME)
        assert result == expected_chunks


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

    def test_no_overlap_with_model(self):
        input_text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
        tokenize_fn = get_tokenizer_fn(MODEL_NAME)
        sentences = split_sentences(input_text)
        sentence_tokens = [len(tokenize_fn(s)) for s in sentences]
        expected_chunks = [
            "This is sentence one. This is sentence two.",
            "This is sentence three. This is sentence four."
        ]
        result = chunk_sentences(input_text, chunk_size=sum(
            sentence_tokens[:2]), sentence_overlap=0, model=MODEL_NAME)
        assert result == expected_chunks

    def test_with_overlap_with_model(self):
        input_text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
        tokenize_fn = get_tokenizer_fn(MODEL_NAME)
        sentences = split_sentences(input_text)
        sentence_tokens = [len(tokenize_fn(s)) for s in sentences]
        expected_chunks = [
            "This is sentence one. This is sentence two.",
            "This is sentence two. This is sentence three.",
            "This is sentence three. This is sentence four."
        ]
        result = chunk_sentences(input_text, chunk_size=sum(
            sentence_tokens[:2]), sentence_overlap=1, model=MODEL_NAME)
        assert result == expected_chunks


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

    def test_no_overlap_with_model(self):
        input_text = "This is a test text with several words to be chunked into smaller pieces"
        tokenize_fn = get_tokenizer_fn(MODEL_NAME)
        token_ids = tokenize_fn(input_text)
        expected = [
            detokenize(token_ids[:8], MODEL_NAME),
            detokenize(token_ids[8:], MODEL_NAME)
        ]
        expected_indices = [0, 0]
        result, result_indices = chunk_texts_with_indices(
            input_text, chunk_size=8, chunk_overlap=0, model=MODEL_NAME)
        assert result == expected
        assert result_indices == expected_indices

    def test_with_overlap_with_model(self):
        input_text = "This is a test text with several words to be chunked"
        tokenize_fn = get_tokenizer_fn(MODEL_NAME)
        token_ids = tokenize_fn(input_text)
        expected = [
            detokenize(token_ids[:8], MODEL_NAME),
            detokenize(token_ids[5:], MODEL_NAME)
        ]
        expected_indices = [0, 0]
        result, result_indices = chunk_texts_with_indices(
            input_text, chunk_size=8, chunk_overlap=3, model=MODEL_NAME)
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

    def test_no_overlap_with_model(self):
        input_text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
        tokenize_fn = get_tokenizer_fn(MODEL_NAME)
        sentences = split_sentences(input_text)
        sentence_tokens = [len(tokenize_fn(s)) for s in sentences]
        expected = [
            "This is sentence one. This is sentence two.",
            "This is sentence three. This is sentence four."
        ]
        expected_indices = [0, 0]
        result, result_indices = chunk_sentences_with_indices(
            input_text, chunk_size=sum(sentence_tokens[:2]), sentence_overlap=0, model=MODEL_NAME)
        assert result == expected
        assert result_indices == expected_indices

    def test_with_overlap_with_model(self):
        input_text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
        tokenize_fn = get_tokenizer_fn(MODEL_NAME)
        sentences = split_sentences(input_text)
        sentence_tokens = [len(tokenize_fn(s)) for s in sentences]
        expected = [
            "This is sentence one. This is sentence two.",
            "This is sentence two. This is sentence three.",
            "This is sentence three. This is sentence four."
        ]
        expected_indices = [0, 0, 0]
        result, result_indices = chunk_sentences_with_indices(
            input_text, chunk_size=sum(sentence_tokens[:2]), sentence_overlap=1, model=MODEL_NAME)
        assert result == expected
        assert result_indices == expected_indices


class TestChunkHeaders:
    def test_chunk_headers_single_doc(self):
        doc = HeaderDocument(
            id="doc1",
            text="Line 1\nLine 2\nLine 3\nLine 4",
            metadata={"doc_index": 1, "parent_header": "Parent"}
        )
        expected = [
            HeaderDocument(
                id="doc1_chunk_0",
                text="Line 1",
                metadata={
                    "header": "Line 1",
                    "parent_header": "Parent",
                    "header_level": 1,
                    "content": "Line 1",
                    "doc_index": 1,
                    "chunk_index": 0,
                    "texts": ["Line 1"],
                    "tokens": 2
                }
            ),
            HeaderDocument(
                id="doc1_chunk_1",
                text="Line 2",
                metadata={
                    "header": "Line 2",
                    "parent_header": "Parent",
                    "header_level": 1,
                    "content": "Line 2",
                    "doc_index": 1,
                    "chunk_index": 1,
                    "texts": ["Line 2"],
                    "tokens": 2
                }
            ),
            HeaderDocument(
                id="doc1_chunk_2",
                text="Line 3",
                metadata={
                    "header": "Line 3",
                    "parent_header": "Parent",
                    "header_level": 1,
                    "content": "Line 3",
                    "doc_index": 1,
                    "chunk_index": 2,
                    "texts": ["Line 3"],
                    "tokens": 2
                }
            ),
            HeaderDocument(
                id="doc1_chunk_3",
                text="Line 4",
                metadata={
                    "header": "Line 4",
                    "parent_header": "Parent",
                    "header_level": 1,
                    "content": "Line 4",
                    "doc_index": 1,
                    "chunk_index": 3,
                    "texts": ["Line 4"],
                    "tokens": 2
                }
            ),
        ]
        result = chunk_headers([doc], max_tokens=2)
        for r, e in zip(result, expected):
            assert r.id == e.id
            assert r.text == e.text
            assert r.metadata == e.metadata
        logger.debug("Test chunk_headers_single_doc passed")

    def test_chunk_headers_with_model(self):
        doc = HeaderDocument(
            id="doc1",
            text="This is sentence one. This is sentence two.\nThis is sentence three. This is an incomplete",
            metadata={"doc_index": 1, "parent_header": "Parent"}
        )
        tokenize_fn = get_tokenizer_fn(MODEL_NAME)
        sentences = split_sentences(doc.text)
        sentence_tokens = [len(tokenize_fn(s)) for s in sentences]
        expected = [
            HeaderDocument(
                id="doc1_chunk_0",
                text="This is sentence one. This is sentence two.",
                metadata={
                    "header": "This is sentence one.",
                    "parent_header": "Parent",
                    "header_level": 1,
                    "content": "This is sentence one. This is sentence two.",
                    "doc_index": 1,
                    "chunk_index": 0,
                    "texts": ["This is sentence one.", "This is sentence two."],
                    "tokens": sum(sentence_tokens[:2])
                }
            ),
            HeaderDocument(
                id="doc1_chunk_1",
                text="This is sentence three. This is an incomplete",
                metadata={
                    "header": "This is sentence three.",
                    "parent_header": "Parent",
                    "header_level": 1,
                    "content": "This is sentence three. This is an incomplete",
                    "doc_index": 1,
                    "chunk_index": 1,
                    "texts": ["This is sentence three.", "This is an incomplete"],
                    "tokens": sum(sentence_tokens[2:])
                }
            )
        ]
        result = chunk_headers([doc], max_tokens=sum(
            sentence_tokens[:2]), model=MODEL_NAME)
        for r, e in zip(result, expected):
            assert r.id == e.id
            assert r.text == e.text
            assert r.metadata == e.metadata
            # Verify each chunk ends with a complete sentence
            chunk_sentences = split_sentences(r.text)
            assert chunk_sentences == r.metadata["texts"], "Chunk does not contain complete sentences"
        logger.debug("Test chunk_headers_with_model passed")


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

    def test_single_string_with_model(self):
        sample = "This is the first sentence. Here is the second one. This is the third and final sentence."
        tokenize_fn = get_tokenizer_fn(MODEL_NAME)
        token_ids = tokenize_fn(sample)
        expected = detokenize(token_ids[:12], MODEL_NAME)
        result = truncate_texts(sample, max_words=12, model=MODEL_NAME)
        assert result == expected

    def test_list_input_with_model(self):
        sample = [
            "First sentence. Second sentence. Third sentence.",
            "Another paragraph. With more text."
        ]
        tokenize_fn = get_tokenizer_fn(MODEL_NAME)
        expected = [
            detokenize(tokenize_fn(sample[0])[:4], MODEL_NAME),
            detokenize(tokenize_fn(sample[1])[:4], MODEL_NAME)
        ]
        result = truncate_texts(sample, max_words=4, model=MODEL_NAME)
        assert result == expected
