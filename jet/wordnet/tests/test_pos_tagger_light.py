import pytest
from typing import List, Union
from unittest.mock import patch, Mock
from jet.wordnet.pos_tagger_light import preprocess_texts, POSTagger, POSItem
import nltk
from nltk.corpus import stopwords

# Mock external dependencies


def mock_clean_newlines(text: str, max_newlines: int) -> str:
    return text.replace("\n\n", "\n")


def mock_clean_punctuations(text: str) -> str:
    return text.replace("!", "").replace(".", "")


def mock_clean_spaces(text: str) -> str:
    return " ".join(text.split())


def mock_clean_string(text: str) -> str:
    return text.strip()


def mock_get_words(text: str) -> List[str]:
    return text.split()


def mock_load_data(file: str) -> dict:
    return {}


def mock_split_sentences(text: str) -> List[str]:
    return [text]


@pytest.fixture(autouse=True)
def setup_nltk():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('corpora/stopwords')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')


class TestPreprocessTexts:
    @patch('jet.wordnet.pos_tagger_light.clean_newlines', side_effect=mock_clean_newlines)
    @patch('jet.wordnet.pos_tagger_light.clean_punctuations', side_effect=mock_clean_punctuations)
    @patch('jet.wordnet.pos_tagger_light.clean_spaces', side_effect=mock_clean_spaces)
    @patch('jet.wordnet.pos_tagger_light.clean_string', side_effect=mock_clean_string)
    @patch('jet.wordnet.pos_tagger_light.get_words', side_effect=mock_get_words)
    def test_preprocess_single_text(self, mock_get_words, mock_clean_string, mock_clean_spaces, mock_clean_punctuations, mock_clean_newlines):
        # Given: A single text with mixed POS tags, punctuation, and stopwords
        input_text: str = "The quick brown fox jumps over the lazy dog!"
        expected: List[str] = ["quick brown fox jumps lazy dog"]
        stop_words: set = set(stopwords.words('english'))

        # When: preprocess_texts is called
        with patch('jet.wordnet.pos_tagger_light.load_data', side_effect=mock_load_data), \
                patch('jet.wordnet.pos_tagger_light.split_sentences', side_effect=mock_split_sentences):
            result: List[str] = preprocess_texts(input_text)

        # Then: The output should be lowercased, cleaned, with only allowed POS tags and no stopwords
        assert result == expected, f"Expected {expected}, but got {result}"

    @patch('jet.wordnet.pos_tagger_light.clean_newlines', side_effect=mock_clean_newlines)
    @patch('jet.wordnet.pos_tagger_light.clean_punctuations', side_effect=mock_clean_punctuations)
    @patch('jet.wordnet.pos_tagger_light.clean_spaces', side_effect=mock_clean_spaces)
    @patch('jet.wordnet.pos_tagger_light.clean_string', side_effect=mock_clean_string)
    @patch('jet.wordnet.pos_tagger_light.get_words', side_effect=mock_get_words)
    def test_preprocess_multiple_texts(self, mock_get_words, mock_clean_string, mock_clean_spaces, mock_clean_punctuations, mock_clean_newlines):
        # Given: A list of texts with various POS tags and stopwords
        input_texts: List[str] = [
            "Dr. Jose Rizal is a hero!",
            "The sun sets slowly behind the mountain."
        ]
        expected: List[str] = [
            "jose rizal hero",
            "sun sets slowly mountain"
        ]

        # When: preprocess_texts is called
        with patch('jet.wordnet.pos_tagger_light.load_data', side_effect=mock_load_data), \
                patch('jet.wordnet.pos_tagger_light.split_sentences', side_effect=mock_split_sentences):
            result: List[str] = preprocess_texts(input_texts)

        # Then: Each text should be processed correctly
        assert result == expected, f"Expected {expected}, but got {result}"

    @patch('jet.wordnet.pos_tagger_light.clean_newlines', side_effect=mock_clean_newlines)
    @patch('jet.wordnet.pos_tagger_light.clean_punctuations', side_effect=mock_clean_punctuations)
    @patch('jet.wordnet.pos_tagger_light.clean_spaces', side_effect=mock_clean_spaces)
    @patch('jet.wordnet.pos_tagger_light.clean_string', side_effect=mock_clean_string)
    @patch('jet.wordnet.pos_tagger_light.get_words', side_effect=mock_get_words)
    def test_preprocess_text_with_newlines(self, mock_get_words, mock_clean_string, mock_clean_spaces, mock_clean_punctuations, mock_clean_newlines):
        # Given: A text with multiple newlines and mixed content
        input_text: str = "Quick foxes climb steep hills.\n\nThe dog sleeps."
        expected: List[str] = ["quick foxes climb steep hills\ndog sleeps"]

        # When: preprocess_texts is called
        with patch('jet.wordnet.pos_tagger_light.load_data', side_effect=mock_load_data), \
                patch('jet.wordnet.pos_tagger_light.split_sentences', side_effect=mock_split_sentences):
            result: List[str] = preprocess_texts(input_text)

        # Then: Newlines should be cleaned, and only allowed POS tags should remain
        assert result == expected, f"Expected {expected}, but got {result}"


class TestPOSTagger:
    @patch('jet.wordnet.pos_tagger_light.load_data', side_effect=mock_load_data)
    @patch('jet.wordnet.pos_tagger_light.split_sentences', side_effect=mock_split_sentences)
    def test_tag_string(self, mock_split_sentences, mock_load_data):
        # Given: A simple sentence and a POS tagger
        input_text: str = "The quick fox jumps"
        tagger: POSTagger = POSTagger()
        expected: List[POSItem] = [
            {'word': 'The', 'pos': 'DET'},
            {'word': 'quick', 'pos': 'ADJ'},
            {'word': 'fox', 'pos': 'NOUN'},
            {'word': 'jumps', 'pos': 'VERB'}
        ]

        # When: tag_string is called
        result: List[POSItem] = tagger.tag_string(input_text)

        # Then: The words should be tagged with correct POS tags
        assert result == expected, f"Expected {expected}, but got {result}"

    @patch('jet.wordnet.pos_tagger_light.load_data', side_effect=mock_load_data)
    @patch('jet.wordnet.pos_tagger_light.split_sentences', side_effect=mock_split_sentences)
    def test_filter_pos(self, mock_split_sentences, mock_load_data):
        # Given: A sentence and a tagger with specific POS includes
        input_text: str = "The quick fox jumps over lazy dog"
        tagger: POSTagger = POSTagger()
        includes_pos: List[str] = ["NOUN", "VERB", "ADJ", "ADV"]
        expected: List[POSItem] = [
            {'word': 'quick', 'pos': 'ADJ'},
            {'word': 'fox', 'pos': 'NOUN'},
            {'word': 'jumps', 'pos': 'VERB'},
            {'word': 'lazy', 'pos': 'ADJ'},
            {'word': 'dog', 'pos': 'NOUN'}
        ]

        # When: filter_pos is called
        result: List[POSItem] = tagger.filter_pos(
            input_text, includes=includes_pos)

        # Then: Only words with specified POS tags should be returned
        assert result == expected, f"Expected {expected}, but got {result}"

    @patch('jet.wordnet.pos_tagger_light.load_data', side_effect=mock_load_data)
    @patch('jet.wordnet.pos_tagger_light.split_sentences', side_effect=mock_split_sentences)
    def test_merge_multi_word_pos(self, mock_split_sentences, mock_load_data):
        # Given: A POS result list with a hyphenated word
        tagger: POSTagger = POSTagger()
        input_pos: List[POSItem] = [
            {'word': 'well', 'pos': 'ADV'},
            {'word': '-', 'pos': 'PUNCT'},
            {'word': 'known', 'pos': 'ADJ'}
        ]
        expected: List[POSItem] = [
            {'word': 'well-known', 'pos': 'ADJ'}
        ]

        # When: merge_multi_word_pos is called
        result: List[POSItem] = tagger.merge_multi_word_pos(input_pos)

        # Then: Hyphenated words should be merged with correct POS
        assert result == expected, f"Expected {expected}, but got {result}"
