import unittest
from nltk.corpus import wordnet
from jet.features.nltk_search import ALLOWED_POS, get_pos_tag, search_by_pos


class TestSearchDocuments(unittest.TestCase):
    def setUp(self):
        self.documents = [
            "The quick brown fox jumps over the lazy dog",
            "A fox fled from danger",
            "The dog sleeps peacefully",
            "Quick foxes climb steep hills"
        ]

    def test_get_pos_tag(self):
        sentence = "The quick foxes run dangerously"
        result = get_pos_tag(sentence)
        # Check if output is a list of (word, pos, lemma) tuples with allowed POS tags
        self.assertIsInstance(result, list)
        self.assertTrue(all(len(item) == 3 for item in result))
        self.assertTrue(all(item[1] in ALLOWED_POS for item in result))
        # Check specific lemmatization and POS tagging
        expected = [
            ('quick', 'JJ', 'quick'),
            ('foxes', 'NNS', 'fox'),
            ('run', 'VBP', 'run'),
            ('dangerously', 'RB', 'dangerously')
        ]
        self.assertEqual(result, expected)

    def test_search_documents_valid_query(self):
        query = "The quick foxes run dangerously"
        results = search_by_pos(query, self.documents)
        # Check result structure
        self.assertIsInstance(results, list)
        self.assertTrue(all(len(item) == 3 for item in results))
        # Check specific results for document 3 (should rank first due to tie-breaker)
        self.assertEqual(results[0][0], 3)  # Document index
        self.assertEqual(results[0][1], 2)  # Match count (quick, fox)
        self.assertEqual(
            results[0][2],  # Matching words with POS and lemma
            [('quick', 'JJ', 'quick'), ('foxes', 'NNS', 'fox')]
        )
        # Check that all matches have allowed POS tags
        self.assertTrue(all(pos in ALLOWED_POS for _, pos, _ in results[0][2]))
        # Check sorting (descending match count, descending index for ties)
        match_counts = [result[1] for result in results]
        self.assertEqual(match_counts, sorted(match_counts, reverse=True))
        # Check tie-breaker: for equal match counts, higher index comes first
        self.assertEqual([result[0] for result in results[:2]], [3, 0])

    def test_search_documents_empty_query(self):
        query = ""
        results = search_by_pos(query, self.documents)
        self.assertEqual(len(results), len(self.documents))
        self.assertTrue(
            all(result[1] == 0 for result in results))  # No matches
        self.assertTrue(all(result[2] == []
                        for result in results))  # Empty match lists

    def test_search_documents_no_matches(self):
        query = "unicorn magical"  # 'magical' is JJ, but no matches in documents
        results = search_by_pos(query, self.documents)
        self.assertTrue(
            all(result[1] == 0 for result in results))  # No matches
        self.assertTrue(all(result[2] == []
                        for result in results))  # Empty match lists

    def test_search_documents_single_word_query(self):
        query = "fox"
        results = search_by_pos(query, self.documents)
        # Documents 0, 1, 3 should have matches for "fox"
        matching_docs = [result[0] for result in results if result[1] > 0]
        self.assertEqual(set(matching_docs), {3, 1, 0})  # Higher indices first
        # Check that matches have allowed POS tags
        for result in results:
            self.assertTrue(all(pos in ALLOWED_POS for _, pos, _ in result[2]))


if __name__ == "__main__":
    # Run unit tests
    unittest.main(argv=[''], exit=False)
