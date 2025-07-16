import unittest
from typing import List
from nltk.corpus import wordnet
from nltk_search import ALLOWED_POS, get_pos_tag, search_by_pos, PosTag, SearchResult


class TestSearchDocuments(unittest.TestCase):
    def setUp(self) -> None:
        self.documents: List[str] = [
            "The quick brown fox jumps over the lazy dog",
            "A fox fled from danger",
            "The dog sleeps peacefully",
            "Quick foxes climb steep hills"
        ]

    def test_get_pos_tag(self) -> None:
        sentence: str = "The quick foxes run dangerously"
        result: List[PosTag] = get_pos_tag(sentence)
        # Check if output is a list of PosTag dictionaries
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(item, dict) for item in result))
        self.assertTrue(
            all('word' in item and 'pos' in item and 'lemma' in item for item in result))
        self.assertTrue(all(item['pos'] in ALLOWED_POS for item in result))
        # Check specific lemmatization and POS tagging
        expected: List[PosTag] = [
            {'word': 'quick', 'pos': 'JJ', 'lemma': 'quick'},
            {'word': 'foxes', 'pos': 'NNS', 'lemma': 'fox'},
            {'word': 'run', 'pos': 'VBP', 'lemma': 'run'},
            {'word': 'dangerously', 'pos': 'RB', 'lemma': 'dangerously'}
        ]
        self.assertEqual(result, expected)

    def test_search_documents_valid_query(self) -> None:
        query: str = "The quick foxes run dangerously"
        results: List[SearchResult] = search_by_pos(query, self.documents)
        self.assertIsInstance(results, list)
        self.assertTrue(all(isinstance(item, dict) for item in results))
        self.assertTrue(all(
            'document_index' in item and 'matching_words_count' in item and 'matching_words_with_pos_and_lemma' in item for item in results))
        self.assertEqual(results[0]['document_index'], 3)
        self.assertEqual(results[0]['matching_words_count'], 2)
        self.assertEqual(
            results[0]['matching_words_with_pos_and_lemma'],
            [{'word': 'quick', 'pos': 'JJ', 'lemma': 'quick'}, {
                'word': 'foxes', 'pos': 'NNS', 'lemma': 'fox'}]
        )
        self.assertTrue(all(
            pos_tag['pos'] in ALLOWED_POS for pos_tag in results[0]['matching_words_with_pos_and_lemma']))
        match_counts = [result['matching_words_count'] for result in results]
        self.assertEqual(match_counts, sorted(match_counts, reverse=True))
        self.assertEqual([result['document_index']
                          # Changed [3, 1, 0] to [3, 0, 1]
                          for result in results[:3]], [3, 0, 1])

    def test_search_documents_empty_query(self) -> None:
        query: str = ""
        results: List[SearchResult] = search_by_pos(query, self.documents)
        self.assertEqual(len(results), len(self.documents))
        self.assertTrue(
            # No matches
            all(result['matching_words_count'] == 0 for result in results))
        self.assertTrue(all(result['matching_words_with_pos_and_lemma'] == [
        ] for result in results))  # Empty match lists

    def test_search_documents_no_matches(self) -> None:
        query: str = "unicorn magical"  # 'magical' is JJ, but no matches in documents
        results: List[SearchResult] = search_by_pos(query, self.documents)
        self.assertTrue(
            # No matches
            all(result['matching_words_count'] == 0 for result in results))
        self.assertTrue(all(result['matching_words_with_pos_and_lemma'] == [
        ] for result in results))  # Empty match lists

    def test_search_documents_single_word_query(self) -> None:
        query: str = "fox"
        results: List[SearchResult] = search_by_pos(query, self.documents)
        # Documents 0, 1, 3 should have matches for "fox"
        matching_docs = [result['document_index']
                         for result in results if result['matching_words_count'] > 0]
        self.assertEqual(set(matching_docs), {0, 1, 3})  # Higher indices first
        # Check that matches have allowed POS tags
        for result in results:
            self.assertTrue(all(
                pos_tag['pos'] in ALLOWED_POS for pos_tag in result['matching_words_with_pos_and_lemma']))


if __name__ == "__main__":
    # Run unit tests
    unittest.main(argv=[''], exit=False)
