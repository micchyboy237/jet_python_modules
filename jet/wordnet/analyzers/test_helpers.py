import unittest
from unittest.mock import patch
from instruction_generator.analyzers.helpers import get_tagalog_words, get_english_words


class TestGetLanguageWords(unittest.TestCase):
    @patch('instruction_generator.analyzers.helpers.load_data')
    def test_get_tagalog_words(self, mock_load_data):
        # Mocking the return value of load_data
        mock_load_data.return_value = {
            "NOUN": {"Aso": 3, "Pusa": 12, "bahay": 3},
            "VERB": {"takbo": 10, "lakad": 2, "aso": 2},
            "ADJ": {"malaki": 8, "maliit": 4},
            "ADV": {"mabilis": 6}
        }

        expected = ['aso', 'pusa', 'takbo', 'malaki', 'mabilis']
        actual = get_tagalog_words(count_threshold=5, includes_pos=[
                                   'NOUN', 'VERB', 'ADJ', 'ADV'])
        self.assertEqual(expected, actual)

    @patch('instruction_generator.analyzers.helpers.load_data')
    def test_get_english_words(self, mock_load_data):
        # Mocking the return value of load_data
        mock_load_data.return_value = {
            "NOUN": {"Dog": 3, "Cat": 12, "house": 3},
            "VERB": {"run": 10, "walk": 2, "dog": 2},
            "ADJ": {"big": 8, "small": 4},
            "ADV": {"fast": 6}
        }

        expected = ['dog', 'cat', 'run', 'big', 'fast']
        actual = get_english_words(count_threshold=5, includes_pos=[
                                   'NOUN', 'VERB', 'ADJ', 'ADV'])
        self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
