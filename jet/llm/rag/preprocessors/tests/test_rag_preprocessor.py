import pytest
from typing import List, Dict, Tuple, TypedDict
from jet.llm.rag.preprocessors.rag_preprocessor import RAGPreprocessor, Chunk
from textblob import TextBlob
from textblob.tokenizers import SentenceTokenizer, WordTokenizer
from textblob.en.np_extractors import ConllExtractor
from textblob.en.taggers import NLTKTagger
from textblob.en.sentiments import PatternAnalyzer


class TestRAGPreprocessor:
    @pytest.fixture
    def preprocessor(self):
        """Fixture for RAGPreprocessor with default settings."""
        return RAGPreprocessor(
            tokenizer=SentenceTokenizer(),
            np_extractor=ConllExtractor(),
            pos_tagger=NLTKTagger(),
            analyzer=PatternAnalyzer()
        )

    def test_create_blob_sentence_tokenizer(self, preprocessor):
        # Given: A sample text and default tokenizer (SentenceTokenizer)
        text = "The AI model is great."
        expected_type = SentenceTokenizer

        # When: We create a blob without word_tokenize
        blob = preprocessor.create_blob(text)

        # Then: The blob should use SentenceTokenizer
        assert isinstance(
            blob.tokenizer, expected_type), f"Expected {expected_type}, got {type(blob.tokenizer)}"

    def test_create_blob_word_tokenizer(self, preprocessor):
        # Given: A sample text and word_tokenize=True
        text = "The AI model is great."
        expected_type = WordTokenizer

        # When: We create a blob with word_tokenize=True
        blob = preprocessor.create_blob(text, word_tokenize=True)

        # Then: The blob should use WordTokenizer
        assert isinstance(
            blob.tokenizer, expected_type), f"Expected {expected_type}, got {type(blob.tokenizer)}"

    def test_preprocess_for_rag_multi_sentence(self, preprocessor):
        # Given: A multi-sentence document
        document = (
            "The new AI model is revolutionary. It processes data quickly. "
            "Developers are excited about its potential."
        )
        expected: List[Chunk] = [
            {
                "text": "The new AI model is revolutionary.",
                "noun_phrases": ["new ai model"],
                "pos_tags": [
                    ("The", "DT"), ("new", "JJ"), ("AI", "NNP"),
                    ("model", "NN"), ("is", "VBZ"), ("revolutionary", "JJ"),
                    (".", ".")
                ],
                "sentiment": {"polarity": 0.5, "subjectivity": 0.8}
            },
            {
                "text": "It processes data quickly.",
                "noun_phrases": ["data"],
                "pos_tags": [
                    ("It", "PRP"), ("processes", "VBZ"), ("data", "NNS"),
                    ("quickly", "RB"), (".", ".")
                ],
                "sentiment": {"polarity": 0.3333333333333333, "subjectivity": 0.6666666666666666}
            },
            {
                "text": "Developers are excited about its potential.",
                "noun_phrases": ["developers", "potential"],
                "pos_tags": [
                    ("Developers", "NNS"), ("are", "VBP"), ("excited", "VBN"),
                    ("about", "IN"), ("its", "PRP$"), ("potential", "NN"),
                    (".", ".")
                ],
                "sentiment": {"polarity": 0.375, "subjectivity": 0.75}
            }
        ]

        # When: We preprocess the document
        result = preprocessor.preprocess_for_rag(document)

        # Then: The chunks should match the expected structure
        assert len(result) == len(
            expected), f"Expected {len(expected)} chunks, got {len(result)}"
        for i, (res, exp) in enumerate(zip(result, expected)):
            assert res["text"] == exp["text"], f"Chunk {i} text mismatch"
            assert res["noun_phrases"] == exp["noun_phrases"], f"Chunk {i} noun phrases mismatch"
            assert res["pos_tags"] == exp["pos_tags"], f"Chunk {i} POS tags mismatch"
            assert res["sentiment"] == exp["sentiment"], f"Chunk {i} sentiment mismatch"

    def test_preprocess_for_rag_single_sentence(self, preprocessor):
        # Given: A single-sentence document
        document = "The AI system performs well."
        expected: List[Chunk] = [
            {
                "text": "The AI system performs well.",
                "noun_phrases": ["ai system"],
                "pos_tags": [
                    ("The", "DT"), ("AI", "NNP"), ("system", "NN"),
                    ("performs", "VBZ"), ("well", "RB"), (".", ".")
                ],
                "sentiment": {"polarity": 0.0, "subjectivity": 0.0}
            }
        ]

        # When: We preprocess the document
        result = preprocessor.preprocess_for_rag(document)

        # Then: The chunk should match the expected structure
        assert len(result) == len(
            expected), f"Expected {len(expected)} chunks, got {len(result)}"
        for i, (res, exp) in enumerate(zip(result, expected)):
            assert res["text"] == exp["text"], f"Chunk {i} text mismatch"
            assert res["noun_phrases"] == exp["noun_phrases"], f"Chunk {i} noun phrases mismatch"
            assert res["pos_tags"] == exp["pos_tags"], f"Chunk {i} POS tags mismatch"
            assert res["sentiment"] == exp["sentiment"], f"Chunk {i} sentiment mismatch"

    def test_preprocess_for_rag_empty_document(self, preprocessor):
        # Given: An empty document
        document = ""
        expected: List[Chunk] = []

        # When: We preprocess the document
        result = preprocessor.preprocess_for_rag(document)

        # Then: The result should be an empty list
        assert result == expected, f"Expected {expected}, got {result}"

    def test_preprocess_for_rag_complex_noun_phrases(self, preprocessor):
        # Given: A document with complex noun phrases
        document = "The advanced machine learning algorithm improves performance."
        expected: List[Chunk] = [
            {
                "text": "The advanced machine learning algorithm improves performance.",
                "noun_phrases": ["advanced machine learning algorithm", "performance"],
                "pos_tags": [
                    ("The", "DT"), ("advanced", "JJ"), ("machine", "NN"),
                    ("learning", "NN"), ("algorithm", "NN"), ("improves", "VBZ"),
                    ("performance", "NN"), (".", ".")
                ],
                "sentiment": {"polarity": 0.0, "subjectivity": 0.0}
            }
        ]

        # When: We preprocess the document
        result = preprocessor.preprocess_for_rag(document)

        # Then: The chunk should match the expected structure
        assert len(result) == len(
            expected), f"Expected {len(expected)} chunks, got {len(result)}"
        for i, (res, exp) in enumerate(zip(result, expected)):
            assert res["text"] == exp["text"], f"Chunk {i} text mismatch"
            assert res["noun_phrases"] == exp["noun_phrases"], f"Chunk {i} noun phrases mismatch"
            assert res["pos_tags"] == exp["pos_tags"], f"Chunk {i} POS tags mismatch"
            assert res["sentiment"] == exp["sentiment"], f"Chunk {i} sentiment mismatch"

    def test_preprocess_for_rag_invalid_input(self, preprocessor):
        # Given: An invalid non-string input
        document = 123

        # When: We attempt to preprocess the document
        # Then: A TypeError should be raised
        with pytest.raises(TypeError, match="The `text` argument passed to `__init__\(text\)` must be a string, not"):
            preprocessor.preprocess_for_rag(document)

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Clean up any resources if needed."""
        yield
        # No specific cleanup required for TextBlob
