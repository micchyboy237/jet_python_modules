import unittest
from unittest.mock import patch
from jet.wordnet.sentence import adaptive_split
from jet.wordnet.pos_tagger import POSTagger
from jet.file.utils import load_data
from collections import defaultdict


class TestPOSTagger(unittest.TestCase):

    def test_singleton_pattern(self):
        # Test Singleton behavior
        instance1 = POSTagger()
        instance2 = POSTagger()
        self.assertIs(instance1, instance2,
                      "POSTagger should follow the Singleton pattern.")

    @patch("jet.file.utils.load_data")  # Ensure we patch the correct path
    def test_initialization_with_dictionary_file(self, mock_load_data):
        # Mock the load_data function to return a sample cache
        sample_cache = {
            'en': {
                'I am learning Python': [
                    {'word': 'I', 'pos': 'PRON'},
                    {'word': 'am', 'pos': 'AUX'},
                    {'word': 'learning', 'pos': 'VERB'},
                    {'word': 'Python', 'pos': 'PROPN'}
                ]
            }
        }
        # Mocking load_data to return sample_cache
        mock_load_data.return_value = sample_cache

        pos_tagger = POSTagger(dictionary_file="dummy_path")

        # Ensure that the dictionary (cache) is initialized properly
        self.assertIsInstance(pos_tagger.cache, defaultdict)
        self.assertEqual(pos_tagger.cache, sample_cache)
        mock_load_data.assert_called_once_with("dummy_path")

    @patch("spacy.load")
    def test_spacy_model_loading(self, mock_spacy_load):
        # Test model loading
        pos_tagger = POSTagger()
        pos_tagger.load_model()

        self.assertIn('en', pos_tagger.nlp_models)

    @patch("jet.wordnet.sentence.adaptive_split")
    @patch("spacy.load")
    def test_text_processing_and_pos_tagging(self, mock_spacy_load, mock_adaptive_split):
        # Test processing and POS tagging
        mock_nlp = mock_spacy_load.return_value
        pos_tagger = POSTagger()

        text = "I am learning Python"
        sentences = ["I am learning Python"]
        mock_adaptive_split.return_value = sentences
        doc_mock = mock_nlp(text)

        # Simulate POS tagging results
        doc_mock.__iter__.return_value = [
            type("Token", (object,), {"text": "I", "pos_": "PRON"}),
            type("Token", (object,), {"text": "am", "pos_": "AUX"}),
            type("Token", (object,), {"text": "learning", "pos_": "VERB"}),
            type("Token", (object,), {"text": "Python", "pos_": "PROPN"})
        ]

        pos_results = pos_tagger.process_and_tag(text)

        # Ensure correct POS tags are assigned to each word
        self.assertEqual(pos_results, [
            {'word': 'I', 'pos': 'PRON'},
            {'word': 'am', 'pos': 'AUX'},
            {'word': 'learning', 'pos': 'VERB'},
            {'word': 'Python', 'pos': 'PROPN'}
        ])

    def test_word_tagging(self):
        # Test tagging a single word
        pos_tagger = POSTagger()
        pos = pos_tagger.tag_word("learning")
        self.assertEqual(pos, "VERB")

    def test_multi_word_pos_merging(self):
        # Test multi-word POS merging
        pos_tagger = POSTagger()

        pos_results = [
            {'word': 'state', 'pos': 'NOUN'},
            {'word': '-', 'pos': '-'},
            {'word': 'of', 'pos': 'ADP'},
            {'word': '-', 'pos': '-'},
            {'word': 'the', 'pos': 'DET'},
            {'word': 'art', 'pos': 'NOUN'},
            {'word': 'technology', 'pos': 'NOUN'}
        ]

        results = pos_tagger.merge_multi_word_pos(pos_results)
        expected = [
            {
                "word": "state-of-the",
                "pos": "NOUN"
            },
            {
                "word": "art",
                "pos": "NOUN"
            },
            {
                "word": "technology",
                "pos": "NOUN"
            }
        ]

        # Verify that hyphenated words are merged
        self.assertEqual(results, expected)

    def test_proper_noun_removal(self):
        # Test proper noun removal
        pos_tagger = POSTagger()
        text = "John is learning Python"
        cleaned_text = pos_tagger.remove_proper_nouns(text)
        self.assertEqual(cleaned_text, "is learning")

    def test_pos_validation(self):
        # Test POS validation for specific words at specific indices
        pos_tagger = POSTagger()
        text = "John is learning Python"
        pos_index_mapping = {
            "0": {"includes": ["PROPN"]},
            "1": {"includes": ["AUX"]},
            "2": {"includes": ["VERB"]},
            "3": {"includes": ["PROPN"]}
        }
        result = pos_tagger.validate_pos(text, pos_index_mapping)
        self.assertTrue(result)

    def test_pos_filtering(self):
        # Test filtering POS tags
        pos_tagger = POSTagger()
        text = "John is learning Python"
        filtered_pos = pos_tagger.filter_pos(text, includes=["VERB"])
        expected = [
            {
                "word": "learning",
                "pos": "VERB"
            }
        ]

        self.assertEqual(filtered_pos, expected)

    def test_word_filtering(self):
        # Test filtering words based on POS tags
        pos_tagger = POSTagger()
        text = "John is learning Python"
        filtered_words = pos_tagger.filter_words(text, includes=["VERB"])

        self.assertEqual(filtered_words, "learning")

    def test_pos_existence_check(self):
        # Test POS existence check
        pos_tagger = POSTagger()
        text = "John is learning Python"
        result = pos_tagger.contains_pos(text, "VERB")
        self.assertTrue(result)

        result = pos_tagger.contains_pos(text, "ADV")
        self.assertFalse(result)

    def test_pos_validation_equal(self):
        # Test validation of equal POS in two texts
        pos_tagger = POSTagger()
        tl_text = "John is learning Python"
        en_text = "John is learning Python"

        result = pos_tagger.validate_equal_pos(tl_text, en_text, "VERB")
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
