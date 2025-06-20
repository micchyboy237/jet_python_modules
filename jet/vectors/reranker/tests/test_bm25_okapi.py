import pytest
from rank_bm25 import BM25Okapi

from jet.vectors.reranker.bm25_okapi import prepare_corpus, tokenize_and_stem


class TestStemming:
    def test_tokenize_and_stem(self):
        text = "Natural language processing processes"
        expected = ["natur", "languag", "process", "process"]
        result = tokenize_and_stem(text)
        assert result == expected

    def test_get_scores_with_stemming(self):
        tokenized_corpus, raw_corpus = prepare_corpus()
        bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75, epsilon=0.25)
        query = tokenize_and_stem("natural language processing text mining")
        scores = bm25.get_scores(query)
        expected = [0.0, 0.3489, 0.0, 0.0, 0.5079]  # Adjust if fixed
        for i, (score, exp) in enumerate(zip(scores, expected)):
            assert score == pytest.approx(
                exp, abs=0.1), f"Mismatch at index {i}"
