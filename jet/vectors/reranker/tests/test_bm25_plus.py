import pytest
import math
from typing import List, Tuple

from jet.vectors.reranker.bm25_plus2 import BM25Plus


@pytest.fixture
def bm25_plus():
    return BM25Plus(k1=1.5, b=0.75, delta=1.0, use_stemming=False)


@pytest.fixture
def sample_docs():
    return [
        "Apple and banana are fruits. Orange is juicy.",
        "Banana grape kiwi smoothie.",
        "Apple kiwi pear dessert."
    ]


class TestBM25Plus:
    def test_preprocess_cleans_text(self, bm25_plus):
        Given: "A raw document text"
        text = "The Apple & Banana are Fruits!"

        When: "We preprocess the text"
        result_tokens = bm25_plus.preprocess(text)

        Then: "Text should be cleaned, lowercased, and stop words removed"
        expected_tokens = ["apple", "banana", "fruits"]
        assert result_tokens == expected_tokens

    def test_fit_calculates_correct_idf_and_avgdl(self, bm25_plus, sample_docs):
        Given: "A BM25+ model and a set of documents"
        doc_tokens = [bm25_plus.preprocess(doc) for doc in sample_docs]
        bm25_plus.fit(doc_tokens)

        When: "We check the IDF values and average document length"
        result_idf = bm25_plus.idf
        result_avgdl = bm25_plus.avgdl

        Then: "IDF and average document length should match expected values"
        expected_idf = {
            "apple": math.log((3 - 2 + 0.5) / (2 + 0.5) + 1),
            "banana": math.log((3 - 2 + 0.5) / (2 + 0.5) + 1),
            "orange": math.log((3 - 1 + 0.5) / (1 + 0.5) + 1),
            "juicy": math.log((3 - 1 + 0.5) / (1 + 0.5) + 1),
            "grape": math.log((3 - 1 + 0.5) / (1 + 0.5) + 1),
            "kiwi": math.log((3 - 2 + 0.5) / (2 + 0.5) + 1),
            "smoothie": math.log((3 - 1 + 0.5) / (1 + 0.5) + 1),
            "pear": math.log((3 - 1 + 0.5) / (1 + 0.5) + 1),
            "dessert": math.log((3 - 1 + 0.5) / (1 + 0.5) + 1),
            "fruits": math.log((3 - 1 + 0.5) / (1 + 0.5) + 1)
        }
        # Corrected: [apple, banana, fruits, orange, juicy], [banana, grape, kiwi, smoothie], [apple, kiwi, pear, dessert]
        expected_avgdl = (5 + 4 + 4) / 3
        assert result_idf == pytest.approx(expected_idf, rel=1e-5)
        assert result_avgdl == pytest.approx(expected_avgdl, rel=1e-5)

    def test_score_ranks_documents_by_relevance(self, bm25_plus, sample_docs):
        Given: "A fitted BM25+ model and a query"
        doc_tokens = [bm25_plus.preprocess(doc) for doc in sample_docs]
        bm25_plus.fit(doc_tokens)
        query = "apple banana"

        When: "We score each document"
        result_scores = [(i, bm25_plus.score(bm25_plus.preprocess(
            query), i, doc)) for i, doc in enumerate(doc_tokens)]

        Then: "Document 0 should have the highest score"
        expected_scores = [(i, bm25_plus.score(bm25_plus.preprocess(
            query), i, doc)) for i, doc in enumerate(doc_tokens)]
        assert result_scores == pytest.approx(expected_scores, rel=1e-5)
        assert result_scores[0][1] > result_scores[1][1] >= result_scores[2][1]

    def test_search_handles_phrases(self, bm25_plus, sample_docs):
        Given: "A fitted BM25+ model and a query with a phrase"
        doc_tokens = [bm25_plus.preprocess(doc) for doc in sample_docs]
        bm25_plus.fit(doc_tokens)
        query = '"apple banana" kiwi'

        When: "We search the documents"
        result_ranked = bm25_plus.search(query, sample_docs)

        Then: "Document 0 should rank highest due to phrase match"
        expected_ranked = sorted(
            [(i, bm25_plus.score(bm25_plus.preprocess(query), i, doc))
             for i, doc in enumerate(doc_tokens)],
            key=lambda x: (-x[1], x[0])
        )
        assert result_ranked == pytest.approx(expected_ranked, rel=1e-5)
        assert result_ranked[0][0] == 0

    def test_search_with_normalization(self, bm25_plus, sample_docs):
        Given: "A fitted BM25+ model and a query"
        doc_tokens = [bm25_plus.preprocess(doc) for doc in sample_docs]
        bm25_plus.fit(doc_tokens)
        query = "apple banana"

        When: "We search with normalization"
        result_ranked = bm25_plus.search(query, sample_docs, normalize=True)

        Then: "Scores should be normalized by the maximum score"
        raw_ranked = bm25_plus.search(query, sample_docs, normalize=False)
        max_score = max(
            score for _, score in raw_ranked) if raw_ranked else 1.0
        expected_ranked = [(i, score / max_score) for i, score in raw_ranked]
        expected_ranked = sorted(expected_ranked, key=lambda x: (-x[1], x[0]))
        assert result_ranked == pytest.approx(expected_ranked, rel=1e-5)
        assert result_ranked[0][1] == pytest.approx(1.0, rel=1e-5)
