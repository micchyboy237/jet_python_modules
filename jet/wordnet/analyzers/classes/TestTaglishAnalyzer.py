import unittest
from instruction_generator.analyzers.classes.TaglishAnalyzer import TaglishAnalyzer


class TestTaglishAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.analyzer = TaglishAnalyzer()
        self.mock_sentences = [
            "pacreate ng account",
            "pagawa ng bahay",
            "magplay ng games",
            "maglaro sa park",
        ]

    def test_analyze_taglish_patterns(self):
        english_counts, tagalog_counts = self.analyzer.analyze_taglish_patterns(
            self.mock_sentences)
        expected_english_counts = {'account': 1, 'games': 1, 'park': 1}
        expected_tagalog_counts = {'bahay': 1, 'maglaro': 1}

        self.assertEqual(english_counts, expected_english_counts)
        self.assertEqual(tagalog_counts, expected_tagalog_counts)

    def test_get_affixed_words(self):
        expected = {
            'pa': ['pacreate'],
            'mag': ['magplay']
        }

        result = self.analyzer.get_affixed_words(self.mock_sentences)
        self.assertEqual(result, expected)

    def test_get_taglish_sentences(self):
        expected = [
            "pacreate ng account",
            "magplay ng games"
        ]

        result = self.analyzer.get_taglish_sentences(self.mock_sentences)
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
