from sentence_transformers import SentenceTransformer
from jet.wordnet.similarity import (
    filter_highest_similarity,
    query_similarity_scores,
    group_similar_texts,
    score_texts_similarity,
    get_similar_texts,
    differences,
    sentence_similarity,
    similars,
    compare_text_pairs,
    has_close_match,
    score_word_placement_similarity,
    has_approximately_same_word_placement,
    are_texts_similar,
    filter_similar_texts,
    filter_different_texts,
    cluster_texts,
)
# from jet.wordnet.spelling import TextComparator
import unittest


class TestGroupSimilarTexts(unittest.TestCase):
    def setUp(self):
        self.sample_texts = [
            "I love programming in Python.",
            "Python is my favorite programming language.",
            "The weather is great today.",
            "It's a sunny and beautiful day.",
            "I enjoy coding in Python.",
            "Machine learning is fascinating.",
            "Artificial Intelligence is evolving rapidly."
        ]

    def test_grouping(self):
        grouped_texts = group_similar_texts(self.sample_texts, threshold=0.7)

        expected = [
            ["I love programming in Python.",
                "Python is my favorite programming language.", "I enjoy coding in Python."],
            ["The weather is great today.", "It's a sunny and beautiful day."],
            ["Machine learning is fascinating."],
            ["Artificial Intelligence is evolving rapidly."]
        ]

        # Verify that each text appears in exactly one group
        results = [text for group in grouped_texts for text in group]
        self.assertEqual(results, expected)

    def test_empty_list(self):
        self.assertEqual(group_similar_texts([]), [])

    def test_single_text(self):
        self.assertEqual(group_similar_texts(
            ["Hello world!"]), [["Hello world!"]])
