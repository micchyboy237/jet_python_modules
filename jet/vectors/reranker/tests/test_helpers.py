import unittest
from jet.vectors.reranker.bm25_helpers import transform_queries_to_ngrams


class TestTransformQueriesToNgrams(unittest.TestCase):

    def test_single_query_with_matching_ngrams(self):
        query = "hello world test example"
        ngrams = {"hello": 1, "world": 1}
        expected = ["hello world"]
        result = transform_queries_to_ngrams(query, ngrams)
        self.assertEqual(result, expected)

    def test_single_query_without_matching_ngrams(self):
        query = "hello world test example"
        ngrams = {"foo": 1, "bar": 1}
        expected = []
        result = transform_queries_to_ngrams(query, ngrams)
        self.assertEqual(result, expected)

    def test_multiple_queries_with_ngrams(self):
        query = ["hello world", "test example"]
        ngrams = {"hello": 1, "test": 1}
        expected = ["hello", "test"]
        result = transform_queries_to_ngrams(query, ngrams)
        self.assertEqual(result, expected)

    def test_empty_query(self):
        query = ""
        ngrams = {"test": 1}
        expected = []
        result = transform_queries_to_ngrams(query, ngrams)
        self.assertEqual(result, expected)

    def test_partial_ngram_matching(self):
        query = "hello world test example"
        ngrams = {"hello": 1, "test": 1}
        expected = ["hello", "test"]
        result = transform_queries_to_ngrams(query, ngrams)
        self.assertEqual(result, expected)

    def test_all_words_are_ngrams(self):
        query = "one two three"
        ngrams = {"one": 1, "two": 1, "three": 1}
        expected = ["one two three"]
        result = transform_queries_to_ngrams(query, ngrams)
        self.assertEqual(result, expected)

    def test_mixed_ngram_and_non_ngram_words(self):
        query = "apple banana orange pear grape"
        ngrams = {"apple": 1, "banana": 1, "grape": 1}
        expected = ["apple banana", "grape"]
        result = transform_queries_to_ngrams(query, ngrams)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
