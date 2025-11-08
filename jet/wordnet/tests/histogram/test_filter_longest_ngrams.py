import unittest
from jet.wordnet.histogram import TextAnalysis


class TestFilterLongestNgrams(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ta = TextAnalysis(
            ["ang iyong mga", "Pwede humingi ng tawad sa", "lahat ng mga"])

    def test_filter_longest_ngrams(self):
        sample_results = [
            {"ngram": "ang iyong mga", "score": 0.0514887973772314},
            {"ngram": "Pwede humingi ng", "score": 0.051121020253108315},
            {"ngram": "Pwede humingi ng tawad", "score": 0.051121020253108315},
            {"ngram": "Pwede humingi ng tawad sa", "score": 0.051121020253108315},
            {"ngram": "humingi ng tawad sa", "score": 0.051121020253108315},
            {"ngram": "ng tawad sa", "score": 0.051121020253108315},
            {"ngram": "lahat ng mga", "score": 0.050385466004862156},
        ]

        actual = self.ta.filter_longest_ngrams(sample_results)
        expected = [
            {'ngram': 'Pwede humingi ng tawad sa', 'score': 0.051121020253108315},
            {'ngram': 'ang iyong mga', 'score': 0.0514887973772314},
            {'ngram': 'lahat ng mga', 'score': 0.050385466004862156}
        ]
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
