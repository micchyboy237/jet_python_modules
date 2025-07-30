import unittest
from unittest.mock import patch
from jet.wordnet.sentence import split_sentences


class TestSplitSentences(unittest.TestCase):
    def test_base_functionality(self):
        """Test that num_sentence=1 preserves base functionality."""
        text = "This is sentence one. This is sentence two. This is sentence three."
        result = split_sentences(text, num_sentence=1)
        expected = [
            "This is sentence one.",
            "This is sentence two.",
            "This is sentence three."
        ]
        self.assertEqual(result, expected)

    def test_combine_two_sentences(self):
        """Test combining two sentences with num_sentence=2."""
        text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
        result = split_sentences(text, num_sentence=2)
        expected = [
            "This is sentence one.\nThis is sentence two.",
            "This is sentence three.\nThis is sentence four."
        ]
        self.assertEqual(result, expected)

    def test_combine_three_sentences(self):
        """Test combining three sentences with num_sentence=3."""
        text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
        result = split_sentences(text, num_sentence=3)
        expected = [
            "This is sentence one.\nThis is sentence two.\nThis is sentence three.",
            "This is sentence four."
        ]
        self.assertEqual(result, expected)

    def test_empty_input(self):
        """Test empty input string."""
        result = split_sentences("", num_sentence=1)
        self.assertEqual(result, [])

    def test_single_sentence(self):
        """Test single sentence input."""
        text = "This is a single sentence."
        result = split_sentences(text, num_sentence=2)
        expected = ["This is a single sentence."]
        self.assertEqual(result, expected)

    def test_list_marker_and_sentence(self):
        """Test handling of list marker followed by a sentence."""
        text = "1. This is a list item. Regular sentence."
        result = split_sentences(text, num_sentence=1)
        expected = ["1. This is a list item.", "Regular sentence."]
        self.assertEqual(result, expected)

    def test_combine_with_list_marker(self):
        """Test combining sentences including a list item with num_sentence=2."""
        text = "1. This is a list item. Regular sentence."
        result = split_sentences(text, num_sentence=2)
        expected = ["1. This is a list item.\nRegular sentence."]
        self.assertEqual(result, expected)

    def test_invalid_combine_count(self):
        """Test handling of invalid num_sentence (e.g., 0 or negative)."""
        text = "This is sentence one. This is sentence two."
        # Update function to handle invalid num_sentence or test expected behavior
        with self.assertRaises(ValueError):
            split_sentences(text, num_sentence=0)
        with self.assertRaises(ValueError):
            split_sentences(text, num_sentence=-1)

    def test_split_sentences_with_abbreviations(self):
        sample = "Dr. Smith lives in the U.S. He works at Acme Inc. He's great."
        expected = ["Dr. Smith lives in the U.S.",
                    "He works at Acme Inc.", "He's great."]
        result = split_sentences(sample)
        self.assertEqual(result, expected)

    def test_split_sentences_with_enumerated_lists(self):
        sample = "1. Apples are red. 2. Bananas are yellow. 3. Grapes are purple."
        expected = [
            "1. Apples are red.",
            "2. Bananas are yellow.",
            "3. Grapes are purple."
        ]
        result = split_sentences(sample)
        self.assertEqual(result, expected)

    def test_treat_lines_as_sentences(self):
        """Test that lines are treated as sentences (one per line)."""
        text = "This is a line without punctuation\nAnother line with punctuation.\nFinal line"
        # Each line should be treated as a sentence, even without punctuation
        expected = [
            "This is a line without punctuation",
            "Another line with punctuation.",
            "Final line"
        ]
        result = split_sentences(text)
        self.assertEqual(result, expected)
