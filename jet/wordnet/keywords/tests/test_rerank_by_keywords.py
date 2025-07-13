import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from typing import List, Optional, Tuple, Union
import uuid
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer
from jet.wordnet.keywords.keyword_extraction import rerank_by_keywords, SimilarityResult, _count_tokens, preprocess_text


class TestRerankByKeywords(unittest.TestCase):
    def setUp(self):
        # Given: A mock KeyBERT model and realistic example texts
        self.mock_model = MagicMock(spec=KeyBERT)
        self.texts = [
            "The latest smartphone features a 6.7-inch AMOLED display, 128GB storage, and a triple-camera system for stunning photos.",
            "Market trends in 2025 show increased demand for sustainable products and rapid growth in e-commerce platforms."
        ]
        self.ids = ["product1", "report1"]
        self.seed_keywords = ["smartphone", "market trends", "sustainability"]
        self.candidates = ["AMOLED display", "sustainable products"]
        self.vectorizer = CountVectorizer(ngram_range=(1, 3))
        self.mock_keywords = [
            [("AMOLED display", 0.85), ("triple-camera", 0.65)],
            [("sustainable products", 0.75), ("e-commerce", 0.55)]
        ]

    def test_preprocess_text(self):
        # Given: A text with extra whitespace and multiple punctuation
        input_text = "This   smartphone has  a  6.7-inch   display!!! and 128GB  storage..."
        expected = "This smartphone has a 6.7-inch display ! and 128GB storage ."

        # When: The text is preprocessed
        result = preprocess_text(input_text)

        # Then: The text should be cleaned with normalized spaces and punctuation
        self.assertEqual(result, expected)

    def test_rerank_single_text(self):
        # Given: A single product description text and mock keywords
        self.mock_model.extract_keywords.return_value = [self.mock_keywords[0]]
        expected = {
            "id": "product1",
            "rank": 1,
            "doc_index": 0,
            "score": 0.85,
            "text": self.texts[0],
            "tokens": _count_tokens(self.texts[0]),
            "keywords": [
                {"text": "AMOLED display", "score": 0.85},
                {"text": "triple-camera", "score": 0.65}
            ]
        }

        # When: Reranking is performed on a single text
        result = rerank_by_keywords(
            texts=[self.texts[0]],
            keybert_model=self.mock_model,
            ids=["product1"],
            top_n=2
        )

        # Then: The result should match the expected output with correct keywords and score
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], expected)

    def test_rerank_multiple_texts_with_ids(self):
        # Given: Multiple texts (product description and market report) with IDs
        self.mock_model.extract_keywords.return_value = self.mock_keywords
        expected_results = [
            {
                "id": "product1",
                "rank": 1,
                "doc_index": 0,
                "score": 0.85,
                "text": self.texts[0],
                "tokens": _count_tokens(self.texts[0]),
                "keywords": [
                    {"text": "AMOLED display", "score": 0.85},
                    {"text": "triple-camera", "score": 0.65}
                ]
            },
            {
                "id": "report1",
                "rank": 2,
                "doc_index": 1,
                "score": 0.75,
                "text": self.texts[1],
                "tokens": _count_tokens(self.texts[1]),
                "keywords": [
                    {"text": "sustainable products", "score": 0.75},
                    {"text": "e-commerce", "score": 0.55}
                ]
            }
        ]

        # When: Reranking is performed on multiple texts with IDs
        result = rerank_by_keywords(
            texts=self.texts,
            keybert_model=self.mock_model,
            ids=self.ids,
            top_n=2
        )

        # Then: Results should be sorted by score with correct keywords and original texts
        self.assertEqual(len(result), 2)
        for i, res in enumerate(result):
            self.assertEqual(res, expected_results[i])

    def test_rerank_multiple_texts_no_ids(self):
        # Given: Multiple texts without provided IDs and mock keywords
        self.mock_model.extract_keywords.return_value = self.mock_keywords

        # When: Reranking is performed without IDs
        result = rerank_by_keywords(
            texts=self.texts,
            keybert_model=self.mock_model,
            top_n=2
        )

        # Then: Results should have generated UUIDs, correct ranks, and scores
        self.assertEqual(len(result), 2)
        for res in result:
            self.assertTrue(isinstance(res["id"], str))
            self.assertTrue(len(res["id"]) > 0)
        self.assertEqual(result[0]["rank"], 1)
        self.assertEqual(result[0]["score"], 0.85)
        self.assertEqual(result[1]["rank"], 2)
        self.assertEqual(result[1]["score"], 0.75)

    def test_rerank_with_candidates(self):
        # Given: Texts with candidate keywords for extraction
        self.mock_model.extract_keywords.return_value = self.mock_keywords

        # When: Reranking is performed with candidate keywords
        result = rerank_by_keywords(
            texts=self.texts,
            keybert_model=self.mock_model,
            ids=self.ids,
            candidates=self.candidates,
            top_n=2
        )

        # Then: The function should use candidates and return expected keywords
        self.mock_model.extract_keywords.assert_called_with(
            docs=[preprocess_text(text) for text in self.texts],
            candidates=self.candidates,
            seed_keywords=None,
            top_n=2,
            keyphrase_ngram_range=(1, 2),
            stop_words="english"
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["keywords"], [
            {"text": "AMOLED display", "score": 0.85},
            {"text": "triple-camera", "score": 0.65}
        ])

    def test_rerank_with_custom_vectorizer(self):
        # Given: Texts with a custom vectorizer for keyword extraction
        self.mock_model.extract_keywords.return_value = self.mock_keywords

        # When: Reranking is performed with a custom vectorizer
        result = rerank_by_keywords(
            texts=self.texts,
            keybert_model=self.mock_model,
            ids=self.ids,
            vectorizer=self.vectorizer,
            top_n=2
        )

        # Then: The function should use the vectorizer and return expected keywords
        self.mock_model.extract_keywords.assert_called_with(
            docs=[preprocess_text(text) for text in self.texts],
            seed_keywords=None,
            vectorizer=self.vectorizer,
            top_n=2
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["keywords"], [
            {"text": "AMOLED display", "score": 0.85},
            {"text": "triple-camera", "score": 0.65}
        ])

    @patch("jet.wordnet.keywords.keyword_extraction.generate_embeddings")
    def test_rerank_with_embeddings(self, mock_generate_embeddings):
        # Given: Texts with mock embeddings and keywords
        mock_generate_embeddings.side_effect = [
            np.array([[0.1, 0.2], [0.3, 0.4]]),
            np.array([[0.5, 0.6], [0.7, 0.8]])
        ]
        self.mock_model.extract_keywords.return_value = self.mock_keywords

        # When: Reranking is performed with embeddings
        result = rerank_by_keywords(
            texts=self.texts,
            keybert_model=self.mock_model,
            ids=self.ids,
            use_embeddings=True,
            top_n=2,
            keyphrase_ngram_range=(1, 3)
        )

        # Then: Results should include correct scores and keywords
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["score"], 0.85)
        self.assertEqual(result[0]["keywords"], [
            {"text": "AMOLED display", "score": 0.85},
            {"text": "triple-camera", "score": 0.65}
        ])
        mock_generate_embeddings.assert_called()

    def test_empty_input(self):
        # Given: An empty text list
        # When: Reranking is performed with no texts
        result = rerank_by_keywords(
            texts=[],
            keybert_model=self.mock_model
        )

        # Then: The result should be an empty list
        self.assertEqual(result, [])

    def test_no_keywords_returned(self):
        # Given: Texts with no keywords extracted
        self.mock_model.extract_keywords.return_value = [
            [] for _ in self.texts]

        # When: Reranking is performed
        result = rerank_by_keywords(
            texts=self.texts,
            keybert_model=self.mock_model,
            ids=self.ids,
            top_n=2
        )

        # Then: Results should have zero scores and empty keyword lists
        self.assertEqual(len(result), 2)
        for res in result:
            self.assertEqual(res["score"], 0.0)
            self.assertEqual(res["keywords"], [])
            self.assertEqual(res["tokens"], _count_tokens(res["text"]))

    def test_invalid_diversity(self):
        # Given: An invalid diversity value for MMR
        # When: Reranking is attempted with use_mmr=True and invalid diversity
        # Then: A ValueError should be raised
        with self.assertRaises(ValueError):
            rerank_by_keywords(
                texts=self.texts,
                keybert_model=self.mock_model,
                use_mmr=True,
                diversity=1.5
            )

    def test_rerank_with_min_count(self):
        # Given: A text with repeated keywords and a min_count requirement
        self.mock_model.extract_keywords.return_value = [
            [("AMOLED display", 0.85)],
            [("AMOLED display", 0.85)]
        ]
        texts = [
            "The smartphone has an AMOLED display and triple-camera.",
            "The smartphone also features an AMOLED display."
        ]
        ids = ["product1", "product2"]
        expected = [
            {
                "id": "product1",
                "rank": 1,
                "doc_index": 0,
                "score": 0.85,
                "text": texts[0],
                "tokens": _count_tokens(texts[0]),
                "keywords": [
                    {"text": "AMOLED display", "score": 0.85}
                ]
            },
            {
                "id": "product2",
                "rank": 2,
                "doc_index": 1,
                "score": 0.85,
                "text": texts[1],
                "tokens": _count_tokens(texts[1]),
                "keywords": [
                    {"text": "AMOLED display", "score": 0.85}
                ]
            }
        ]

        # When: Reranking with min_count=2 to filter out keywords appearing only once
        mock_vectorizer = MagicMock()
        mock_vectorizer.fit_transform.return_value = np.array(
            [[2, 1, 0], [1, 0, 0]])
        mock_vectorizer.get_feature_names_out.return_value = [
            "AMOLED display", "triple-camera", "e-commerce"]
        with patch('sklearn.feature_extraction.text.CountVectorizer', return_value=mock_vectorizer):
            result = rerank_by_keywords(
                texts=texts,
                keybert_model=self.mock_model,
                ids=ids,
                top_n=2,
                min_count=2
            )

        # Then: Only keywords with count >= 2 (e.g., AMOLED display) should appear
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["keywords"], [
                         {"text": "AMOLED display", "score": 0.85}])
        self.assertEqual(result[1]["keywords"], [
                         {"text": "AMOLED display", "score": 0.85}])

    def test_rerank_with_candidates_and_min_count(self):
        # Given: Texts and candidates with some meeting min_count
        self.mock_model.extract_keywords.return_value = [
            [("AMOLED display", 0.85)],
            [("AMOLED display", 0.75)]
        ]
        texts = [
            "The smartphone has an AMOLED display and triple-camera.",
            "Sustainable products are trending with AMOLED display."
        ]
        ids = ["product1", "report1"]
        candidates = ["AMOLED display",
                      "sustainable products", "triple-camera"]
        expected = [
            {
                "id": "product1",
                "rank": 1,
                "doc_index": 0,
                "score": 0.85,
                "text": texts[0],
                "tokens": _count_tokens(texts[0]),
                "keywords": [
                    {"text": "AMOLED display", "score": 0.85}
                ]
            },
            {
                "id": "report1",
                "rank": 2,
                "doc_index": 1,
                "score": 0.75,
                "text": texts[1],
                "tokens": _count_tokens(texts[1]),
                "keywords": [
                    {"text": "AMOLED display", "score": 0.75}
                ]
            }
        ]

        # When: Reranking with candidates and min_count=2
        mock_vectorizer = MagicMock()
        mock_vectorizer.fit_transform.return_value = np.array(
            [[2, 0, 1], [1, 1, 0]])
        mock_vectorizer.get_feature_names_out.return_value = [
            "AMOLED display", "sustainable products", "triple-camera"]
        with patch('sklearn.feature_extraction.text.CountVectorizer', return_value=mock_vectorizer):
            result = rerank_by_keywords(
                texts=texts,
                keybert_model=self.mock_model,
                ids=ids,
                candidates=candidates,
                top_n=2,
                min_count=2
            )

        # Then: Only AMOLED display should be used as a candidate
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["keywords"], [
                         {"text": "AMOLED display", "score": 0.85}])
        self.assertEqual(result[1]["keywords"], [
                         {"text": "AMOLED display", "score": 0.75}])
