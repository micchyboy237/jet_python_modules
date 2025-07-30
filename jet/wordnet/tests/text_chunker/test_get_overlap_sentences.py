import unittest
from typing import List, Tuple
from unittest.mock import Mock
from jet.wordnet.text_chunker import get_overlap_sentences
from jet.models.tokenizer.base import get_tokenizer_fn
from jet.wordnet.words import get_words


class TestGetOverlapSentences(unittest.TestCase):
    def setUp(self):
        """Set up common test data and functions."""
        self.word_size_fn = get_words
        # Mock token_size_fn to return a list of tokens, mimicking get_words
        self.token_size_fn = Mock()
        self.token_size_fn.side_effect = get_words  # Return list of words directly
        self.sentences = [
            "First sentence.",
            "Second sentence is longer.",
            "Third short."
        ]
        self.separators = [" ", " ", " "]
        # Word counts: ["First", "sentence"] (2), ["Second", "sentence", "is", "longer"] (4), ["Third", "short"] (2)

    def test_basic_overlap(self):
        """Test basic overlap selection with word-based size function."""
        max_overlap = 3
        expected_sentences = ["Third short."]
        expected_separators = [" "]
        expected_size = 2
        result_sentences, result_separators, result_size = get_overlap_sentences(
            self.sentences, self.separators, max_overlap, self.word_size_fn
        )
        self.assertEqual(result_sentences, expected_sentences)
        self.assertEqual(result_separators, expected_separators)
        self.assertEqual(result_size, expected_size)

    def test_zero_overlap(self):
        """Test when max_overlap is 0."""
        max_overlap = 0
        expected_sentences = []
        expected_separators = []
        expected_size = 0
        result_sentences, result_separators, result_size = get_overlap_sentences(
            self.sentences, self.separators, max_overlap, self.word_size_fn
        )
        self.assertEqual(result_sentences, expected_sentences)
        self.assertEqual(result_separators, expected_separators)
        self.assertEqual(result_size, expected_size)

    def test_empty_input(self):
        """Test with empty sentences and separators."""
        max_overlap = 5
        expected_sentences = []
        expected_separators = []
        expected_size = 0
        result_sentences, result_separators, result_size = get_overlap_sentences(
            [], [], max_overlap, self.word_size_fn
        )
        self.assertEqual(result_sentences, expected_sentences)
        self.assertEqual(result_separators, expected_separators)
        self.assertEqual(result_size, expected_size)

    def test_single_sentence_fits(self):
        """Test when a single sentence fits within max_overlap."""
        max_overlap = 5
        sentences = ["Short sentence."]
        separators = [" "]
        expected_sentences = ["Short sentence."]
        expected_separators = [" "]
        expected_size = 2  # "Short", "sentence"
        result_sentences, result_separators, result_size = get_overlap_sentences(
            sentences, separators, max_overlap, self.word_size_fn
        )
        self.assertEqual(result_sentences, expected_sentences)
        self.assertEqual(result_separators, expected_separators)
        self.assertEqual(result_size, expected_size)

    def test_single_sentence_exceeds(self):
        """Test when a single sentence exceeds max_overlap."""
        max_overlap = 2
        sentences = ["Very long sentence here."]
        separators = [" "]
        expected_sentences = []
        expected_separators = []
        expected_size = 0
        result_sentences, result_separators, result_size = get_overlap_sentences(
            sentences, separators, max_overlap, self.word_size_fn
        )
        self.assertEqual(result_sentences, expected_sentences)
        self.assertEqual(result_separators, expected_separators)
        self.assertEqual(result_size, expected_size)

    def test_multiple_sentences_partial_fit(self):
        """Test when multiple sentences are selected but only some fit within max_overlap."""
        max_overlap = 5
        expected_sentences = ["Third short."]
        expected_separators = [" "]
        # Only "Third short." (2 words) fits within max_overlap=5
        expected_size = 2
        result_sentences, result_separators, result_size = get_overlap_sentences(
            self.sentences, self.separators, max_overlap, self.word_size_fn
        )
        self.assertEqual(result_sentences, expected_sentences)
        self.assertEqual(result_separators, expected_separators)
        self.assertEqual(result_size, expected_size)

    def test_token_based_size_fn(self):
        """Test overlap with token-based size function (mocked)."""
        max_overlap = 3
        expected_sentences = ["Third short."]
        expected_separators = [" "]
        expected_size = 2  # Mocked to match word count for "Third short."
        result_sentences, result_separators, result_size = get_overlap_sentences(
            self.sentences, self.separators, max_overlap, self.token_size_fn
        )
        self.assertEqual(result_sentences, expected_sentences)
        self.assertEqual(result_separators, expected_separators)
        self.assertEqual(result_size, expected_size)
        self.token_size_fn.assert_any_call("Third short.")

    def test_token_based_size_fn_call_order(self):
        """Test that token_size_fn is called in the correct (reverse) order."""
        max_overlap = 3
        get_overlap_sentences(
            self.sentences, self.separators, max_overlap, self.token_size_fn
        )
        expected_calls = [
            unittest.mock.call("Third short."),
            unittest.mock.call("Second sentence is longer.")
        ]
        self.token_size_fn.assert_has_calls(expected_calls, any_order=False)

    def test_mismatched_sentences_separators(self):
        """Test when sentences and separators lists have mismatched lengths."""
        max_overlap = 5
        sentences = ["First sentence."]
        separators = []  # Mismatched length
        with self.assertRaises(IndexError):
            get_overlap_sentences(sentences, separators,
                                  max_overlap, self.word_size_fn)


if __name__ == '__main__':
    unittest.main()
