import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from typing import List, Optional, Tuple, Union
import uuid
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer
from jet.wordnet.keywords.keyword_extraction import rerank_by_keywords, SimilarityResult, _count_tokens


class TestRerankByKeywords(unittest.TestCase):
    def setUp(self):
        # Mock KeyBERT model
        self.mock_model = MagicMock(spec=KeyBERT)
        self.texts = [
            "This is a test document about isekai anime.",
            "Another document discussing anime trends in 2025."
        ]
        self.ids = ["doc1", "doc2"]
        self.seed_keywords = ["isekai", "anime", "2025"]
        self.candidates = ["isekai anime", "anime trends"]
        self.vectorizer = CountVectorizer(ngram_range=(1, 3))
        self.mock_keywords = [
            [("isekai anime", 0.8), ("anime", 0.6)],
            [("anime trends", 0.7), ("2025", 0.5)]
        ]

    def test_rerank_single_text(self):
        self.mock_model.extract_keywords.return_value = [
            self.mock_keywords[0]]  # Return list of list of tuples
        result = rerank_by_keywords(
            texts=[self.texts[0]],
            keybert_model=self.mock_model,
            ids=["doc1"],
            top_n=2
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "doc1")
        self.assertEqual(result[0]["rank"], 1)
        self.assertEqual(result[0]["doc_index"], 0)
        self.assertEqual(result[0]["score"], 0.8)
        self.assertEqual(result[0]["text"], self.texts[0])
        self.assertEqual(result[0]["tokens"], _count_tokens(self.texts[0]))
        self.assertEqual(result[0]["keywords"], [
            {"text": "isekai anime", "score": 0.8}, {
                "text": "anime", "score": 0.6}
        ])

    def test_rerank_multiple_texts_with_ids(self):
        self.mock_model.extract_keywords.return_value = self.mock_keywords
        result = rerank_by_keywords(
            texts=self.texts,
            keybert_model=self.mock_model,
            ids=self.ids,
            top_n=2
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["id"], "doc1")  # Highest score: 0.8
        self.assertEqual(result[0]["rank"], 1)
        self.assertEqual(result[0]["score"], 0.8)
        self.assertEqual(result[1]["id"], "doc2")  # Score: 0.7
        self.assertEqual(result[1]["rank"], 2)
        self.assertEqual(result[1]["score"], 0.7)
        for i, res in enumerate(result):
            self.assertEqual(res["doc_index"], i)
            self.assertEqual(res["text"], self.texts[i])
            self.assertEqual(res["tokens"], _count_tokens(self.texts[i]))
            self.assertEqual(res["keywords"], [
                             {"text": kw, "score": score} for kw, score in self.mock_keywords[i]])

    def test_rerank_multiple_texts_no_ids(self):
        self.mock_model.extract_keywords.return_value = self.mock_keywords
        result = rerank_by_keywords(
            texts=self.texts,
            keybert_model=self.mock_model,
            top_n=2
        )
        self.assertEqual(len(result), 2)
        for res in result:
            self.assertTrue(isinstance(res["id"], str))
            self.assertTrue(len(res["id"]) > 0)  # UUID generated
        self.assertEqual(result[0]["rank"], 1)
        self.assertEqual(result[0]["score"], 0.8)
        self.assertEqual(result[1]["rank"], 2)
        self.assertEqual(result[1]["score"], 0.7)

    def test_rerank_with_candidates(self):
        self.mock_model.extract_keywords.return_value = self.mock_keywords
        result = rerank_by_keywords(
            texts=self.texts,
            keybert_model=self.mock_model,
            ids=self.ids,
            candidates=self.candidates,
            top_n=2
        )
        self.mock_model.extract_keywords.assert_called_with(
            docs=self.texts,
            candidates=self.candidates,
            seed_keywords=None,
            top_n=2,
            keyphrase_ngram_range=(1, 2),
            stop_words="english"
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["keywords"], [
                         {"text": "isekai anime", "score": 0.8}, {"text": "anime", "score": 0.6}])

    def test_rerank_with_custom_vectorizer(self):
        self.mock_model.extract_keywords.return_value = self.mock_keywords
        result = rerank_by_keywords(
            texts=self.texts,
            keybert_model=self.mock_model,
            ids=self.ids,
            vectorizer=self.vectorizer,
            top_n=2
        )
        self.mock_model.extract_keywords.assert_called_with(
            docs=self.texts,
            seed_keywords=None,
            vectorizer=self.vectorizer,
            top_n=2
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["keywords"], [
                         {"text": "isekai anime", "score": 0.8}, {"text": "anime", "score": 0.6}])

    @patch("jet.wordnet.keywords.keyword_extraction.generate_embeddings")
    def test_rerank_with_embeddings(self, mock_generate_embeddings):
        mock_generate_embeddings.side_effect = [
            np.array([[0.1, 0.2], [0.3, 0.4]]),  # doc_embeddings
            np.array([[0.5, 0.6], [0.7, 0.8]])   # word_embeddings
        ]
        self.mock_model.extract_keywords.return_value = self.mock_keywords
        result = rerank_by_keywords(
            texts=self.texts,
            keybert_model=self.mock_model,
            ids=self.ids,
            use_embeddings=True,
            top_n=2,
            keyphrase_ngram_range=(1, 3)
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["score"], 0.8)
        self.assertEqual(result[0]["keywords"], [
                         {"text": "isekai anime", "score": 0.8}, {"text": "anime", "score": 0.6}])
        mock_generate_embeddings.assert_called()

    def test_empty_input(self):
        result = rerank_by_keywords(
            texts=[],
            keybert_model=self.mock_model
        )
        self.assertEqual(result, [])

    def test_no_keywords_returned(self):
        self.mock_model.extract_keywords.return_value = [
            [] for _ in self.texts]
        result = rerank_by_keywords(
            texts=self.texts,
            keybert_model=self.mock_model,
            ids=self.ids,
            top_n=2
        )
        self.assertEqual(len(result), 2)
        for res in result:
            self.assertEqual(res["score"], 0.0)
            self.assertEqual(res["keywords"], [])
            self.assertEqual(res["tokens"], _count_tokens(res["text"]))

    def test_invalid_diversity(self):
        with self.assertRaises(ValueError):
            rerank_by_keywords(
                texts=self.texts,
                keybert_model=self.mock_model,
                use_mmr=True,
                diversity=1.5  # Invalid diversity
            )


if __name__ == "__main__":
    unittest.main()
