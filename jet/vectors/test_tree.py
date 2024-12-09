import unittest
from jet.vectors.tree import (
    tokenize_document,
    vectorize_sentences,
    cluster_sentences,
    evaluate_thoughts,
    generate_hierarchical_summary
)


class TestTreeOfThoughts(unittest.TestCase):
    def setUp(self):
        self.document = """
        Natural language processing is an exciting field. It enables computers to understand human language.
        Applications of NLP include sentiment analysis, machine translation, and more.
        Clustering is a useful technique for grouping similar sentences.
        """
        self.sentences = [
            "Natural language processing is an exciting field.",
            "It enables computers to understand human language.",
            "Applications of NLP include sentiment analysis, machine translation, and more.",
            "Clustering is a useful technique for grouping similar sentences."
        ]
        self.num_clusters = 2

    def test_tokenize_document(self):
        result = tokenize_document(self.document)
        self.assertEqual(result, self.sentences)

    def test_vectorize_sentences(self):
        X, vectorizer = vectorize_sentences(self.sentences)
        self.assertEqual(X.shape[0], len(self.sentences))

    def test_cluster_sentences(self):
        X, _ = vectorize_sentences(self.sentences)
        kmeans = cluster_sentences(X, self.num_clusters)
        self.assertEqual(len(kmeans.labels_), len(self.sentences))

    def test_evaluate_thoughts(self):
        X, _ = vectorize_sentences(self.sentences)
        kmeans = cluster_sentences(X, self.num_clusters)
        thought_tree = evaluate_thoughts(self.sentences, kmeans.labels_)
        self.assertEqual(len(thought_tree), self.num_clusters)

    def test_generate_hierarchical_summary(self):
        thought_tree = [
            {"cluster": 0, "sentences": self.sentences[:2], "summary": " ".join(
                self.sentences[:2])},
            {"cluster": 1, "sentences": self.sentences[2:], "summary": " ".join(
                self.sentences[2:])}
        ]
        summary = generate_hierarchical_summary(thought_tree)
        self.assertIn("summary", summary)
        self.assertIn("thoughts", summary)
        self.assertEqual(len(summary["thoughts"]), 2)


if __name__ == "__main__":
    unittest.main()
