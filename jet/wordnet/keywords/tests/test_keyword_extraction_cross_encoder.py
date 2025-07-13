import unittest
from unittest.mock import patch, MagicMock
from typing import List, Optional, Tuple
import numpy as np
import uuid
from sklearn.feature_extraction.text import CountVectorizer
from jet.wordnet.keywords.keyword_extraction import preprocess_text, _count_tokens
from jet.wordnet.keywords.keyword_extraction_cross_encoder import extract_keywords_cross_encoder, CrossEncoderKeywordResult


class TestExtractKeywordsCrossEncoder(unittest.TestCase):
    def setUp(self):
        self.texts = [
            "The latest smartphone features a 6.7-inch AMOLED display and 128GB storage.",
            "Market trends in 2025 show increased demand for sustainable products."
        ]
        self.ids = ["doc1", "doc2"]
        self.candidates = [
            "AMOLED display", "sustainable products", "128GB storage", "market trends"]
        self.vectorizer = CountVectorizer(ngram_range=(1, 2))
        self.mock_scores = [
            [0.85, 0.65, 0.45, 0.25],  # Scores for doc1
            [0.35, 0.75, 0.15, 0.55]   # Scores for doc2
        ]

    @patch("jet.models.model_registry.transformers.cross_encoder_model_registry.CrossEncoderRegistry.load_model")
    def test_extract_keywords_single_text(self, mock_load_model):
        # Given: A single text with mocked cross-encoder output
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array(self.mock_scores[0])
        mock_load_model.return_value = mock_model
        expected = {
            "id": "doc1",
            "rank": 1,
            "doc_index": 0,
            "score": 0.85,
            "text": self.texts[0],
            "tokens": _count_tokens(self.texts[0]),
            "keywords": [
                {"text": "AMOLED display", "score": 0.85},
                {"text": "sustainable products", "score": 0.65}
            ]
        }

        # When: Extract keywords for a single text
        result = extract_keywords_cross_encoder(
            texts=[self.texts[0]],
            cross_encoder_model="ms-marco-MiniLM-L6-v2",
            ids=["doc1"],
            candidates=self.candidates,
            top_n=2
        )

        # Then: Verify the result matches expected output
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], expected)
        mock_model.predict.assert_called_once()

    @patch("jet.models.model_registry.transformers.cross_encoder_model_registry.CrossEncoderRegistry.load_model")
    def test_extract_keywords_multiple_texts_with_ids(self, mock_load_model):
        # Given: Multiple texts with IDs and mocked cross-encoder output
        mock_model = MagicMock()
        mock_model.predict.side_effect = [
            np.array(scores) for scores in self.mock_scores]
        mock_load_model.return_value = mock_model
        expected_results = [
            {
                "id": "doc1",
                "rank": 1,
                "doc_index": 0,
                "score": 0.85,
                "text": self.texts[0],
                "tokens": _count_tokens(self.texts[0]),
                "keywords": [
                    {"text": "AMOLED display", "score": 0.85},
                    {"text": "sustainable products", "score": 0.65}
                ]
            },
            {
                "id": "doc2",
                "rank": 2,
                "doc_index": 1,
                "score": 0.75,
                "text": self.texts[1],
                "tokens": _count_tokens(self.texts[1]),
                "keywords": [
                    {"text": "sustainable products", "score": 0.75},
                    {"text": "market trends", "score": 0.55}
                ]
            }
        ]

        # When: Extract keywords for multiple texts
        result = extract_keywords_cross_encoder(
            texts=self.texts,
            cross_encoder_model="ms-marco-MiniLM-L6-v2",
            ids=self.ids,
            candidates=self.candidates,
            top_n=2
        )

        # Then: Verify the results match expected output
        self.assertEqual(len(result), 2)
        for i, res in enumerate(result):
            self.assertEqual(res, expected_results[i])
        self.assertEqual(mock_model.predict.call_count, 2)

    @patch("jet.models.model_registry.transformers.cross_encoder_model_registry.CrossEncoderRegistry.load_model")
    def test_extract_keywords_no_ids(self, mock_load_model):
        # Given: Multiple texts without IDs and mocked cross-encoder output
        mock_model = MagicMock()
        mock_model.predict.side_effect = [
            np.array(scores) for scores in self.mock_scores]
        mock_load_model.return_value = mock_model

        # When: Extract keywords without providing IDs
        result = extract_keywords_cross_encoder(
            texts=self.texts,
            cross_encoder_model="ms-marco-MiniLM-L6-v2",
            candidates=self.candidates,
            top_n=2
        )

        # Then: Verify UUIDs are generated and rankings are correct
        self.assertEqual(len(result), 2)
        for res in result:
            self.assertTrue(isinstance(res["id"], str))
            self.assertTrue(len(res["id"]) > 0)
        self.assertEqual(result[0]["rank"], 1)
        self.assertEqual(result[0]["score"], 0.85)
        self.assertEqual(result[1]["rank"], 2)
        self.assertEqual(result[1]["score"], 0.75)

    @patch("jet.models.model_registry.transformers.cross_encoder_model_registry.CrossEncoderRegistry.load_model")
    def test_extract_keywords_with_vectorizer(self, mock_load_model):
        # Given: Texts with a custom vectorizer and mocked cross-encoder output
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array(self.mock_scores[0])
        mock_load_model.return_value = mock_model
        with patch("jet.wordnet.keywords.keyword_extraction_cross_encoder.CountVectorizer.fit") as mock_vectorizer_fit:
            mock_vectorizer_fit.return_value.get_feature_names_out.return_value = self.candidates

            # When: Extract keywords using a custom vectorizer
            result = extract_keywords_cross_encoder(
                texts=[self.texts[0]],
                cross_encoder_model="ms-marco-MiniLM-L6-v2",
                vectorizer=self.vectorizer,
                top_n=2
            )

            # Then: Verify vectorizer was used and results are correct
            mock_vectorizer_fit.assert_called_once()
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["keywords"], [
                {"text": "AMOLED display", "score": 0.85},
                {"text": "sustainable products", "score": 0.65}
            ])

    def test_empty_input(self):
        # Given: An empty text list
        # When: Extract keywords from empty input
        result = extract_keywords_cross_encoder(
            texts=[],
            cross_encoder_model="ms-marco-MiniLM-L6-v2"
        )

        # Then: Verify empty result is returned
        self.assertEqual(result, [])

    @patch("jet.models.model_registry.transformers.cross_encoder_model_registry.CrossEncoderRegistry.load_model")
    def test_no_candidates(self, mock_load_model):
        # Given: Texts with a vectorizer that produces no candidates
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        with patch("jet.wordnet.keywords.keyword_extraction_cross_encoder.CountVectorizer.fit") as mock_vectorizer_fit:
            mock_vectorizer_fit.return_value.get_feature_names_out.return_value = []

            # When: Extract keywords with no candidates
            result = extract_keywords_cross_encoder(
                texts=self.texts,
                cross_encoder_model="ms-marco-MiniLM-L6-v2",
                vectorizer=self.vectorizer
            )

            # Then: Verify empty results are returned
            self.assertEqual(result, [])

    @patch("jet.models.model_registry.transformers.cross_encoder_model_registry.CrossEncoderRegistry.load_model")
    def test_cross_encoder_error(self, mock_load_model):
        # Given: A cross-encoder that raises an error
        mock_model = MagicMock()
        mock_model.predict.side_effect = Exception("Prediction error")
        mock_load_model.return_value = mock_model

        # When: Extract keywords with failing cross-encoder
        result = extract_keywords_cross_encoder(
            texts=[self.texts[0]],
            cross_encoder_model="ms-marco-MiniLM-L6-v2",
            candidates=self.candidates,
            top_n=2
        )

        # Then: Verify keywords are returned with zero scores
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["score"], 0.0)
        self.assertEqual(len(result[0]["keywords"]), 2)
        for kw in result[0]["keywords"]:
            self.assertEqual(kw["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
