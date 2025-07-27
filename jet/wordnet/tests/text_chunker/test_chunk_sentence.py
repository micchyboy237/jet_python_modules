from nltk.tokenize import sent_tokenize, word_tokenize
from jet.models.model_types import ModelType
from jet.wordnet.sentence import split_sentences
from jet.wordnet.text_chunker import chunk_sentences
from jet.models.tokenizer.base import get_tokenizer_fn

MODEL_NAME: ModelType = "qwen3-1.7b-4bit"


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
