from typing import List
import pytest
from jet.wordnet.phrase_detector import PhraseDetector, DetectedPhrase, QueryPhraseResult
import os


class TestPhraseDetector:
    @pytest.fixture
    def setup_detector(self, tmp_path):
        # Given: A temporary model path and sample sentences
        model_path = tmp_path / "test_phrase_model.pkl"
        sentences = [
            "Artificial intelligence is transforming industries",
            "Artificial intelligence and machine learning are key technologies",
            "Cloud computing supports machine learning"
        ]
        detector = PhraseDetector(
            model_path=str(model_path),
            sentences=sentences,
            min_count=2,
            threshold=0.3,
            reset_cache=True
        )
        yield detector, sentences, model_path
        # Cleanup handled by tmp_path

    def test_detect_phrases(self, setup_detector):
        detector, texts, _ = setup_detector
        expected = [
            {
                "index": 0,
                "sentence": "Artificial intelligence is transforming industries",
                "phrases": ["artificial intelligence"],
                "results": [
                    {"phrase": "artificial intelligence",
                        "score": pytest.approx(0.6, abs=0.1)}
                ]
            },
            {
                "index": 1,
                "sentence": "Artificial intelligence and machine learning are key technologies",
                "phrases": ["artificial intelligence", "key technologies", "machine learning"],
                "results": [
                    {"phrase": "artificial intelligence",
                        "score": pytest.approx(0.6, abs=0.1)},
                    {"phrase": "key technologies",
                        "score": pytest.approx(0.6, abs=0.1)},
                    {"phrase": "machine learning",
                        "score": pytest.approx(0.6, abs=0.1)}
                ]
            },
            {
                "index": 2,
                "sentence": "Cloud computing supports machine learning",
                "phrases": ["cloud computing", "machine learning"],
                "results": [
                    {"phrase": "cloud computing",
                        "score": pytest.approx(0.7, abs=0.1)},
                    {"phrase": "machine learning",
                        "score": pytest.approx(0.6, abs=0.1)}
                ]
            }
        ]
        result = list(detector.detect_phrases(texts))
        for res, exp in zip(result, expected):
            assert res["index"] == exp["index"], f"Expected index {exp['index']}, got {res['index']}"
            assert res["sentence"] == exp[
                "sentence"], f"Expected sentence {exp['sentence']}, got {res['sentence']}"
            assert res["phrases"] == exp[
                "phrases"], f"Expected phrases {exp['phrases']}, got {res['phrases']}"
            assert len(res["results"]) == len(
                exp["results"]), f"Expected {len(exp['results'])}, got {len(res['results'])}"
            for r, e in zip(res["results"], exp["results"]):
                assert r["phrase"] == e["phrase"], f"Expected phrase {e['phrase']}, got {r['phrase']}"
                assert r["score"] == pytest.approx(
                    e["score"], abs=0.1), f"Expected score {e['score']}, got {r['score']}"

    def test_extract_phrases(self, setup_detector):
        # Given: A detector and test texts
        detector, texts, _ = setup_detector
        expected = [
            "artificial intelligence",
            "cloud computing",
            "key technologies",
            "machine learning"
        ]

        # When: Extracting phrases
        result = detector.extract_phrases(texts)

        # Then: Verify extracted phrases
        assert result == expected, f"Expected phrases {expected}, got {result}"

    def test_get_phrase_grams(self, setup_detector):
        # Given: A detector with prebuilt phrase grams
        detector, _, _ = setup_detector
        expected = {
            "cloud computing": pytest.approx(0.7, abs=0.1),
            "artificial intelligence": pytest.approx(0.6, abs=0.1),
            "key technologies": pytest.approx(0.6, abs=0.1),
            "machine learning": pytest.approx(0.6, abs=0.1),
        }

        # When: Getting phrase grams with a threshold
        result = detector.get_phrase_grams(threshold=0.3)

        # Then: Verify phrase grams and scores
        assert len(result) == len(
            expected), f"Expected {len(expected)} phrases, got {len(result)}"
        for phrase, score in expected.items():
            assert phrase in result, f"Expected phrase {phrase} not in result"
            assert result[phrase] == score, f"Expected score {score} for {phrase}, got {result[phrase]}"

    def test_query(self, setup_detector):
        # Given: A detector and sample queries
        detector, _, _ = setup_detector
        queries = ["machine learning", "cloud"]
        expected = [
            {
                "query": "cloud",
                "phrase": "cloud computing",
                "score": pytest.approx(0.7, abs=0.1)
            },
            {
                "query": "machine_learning",
                "phrase": "machine learning",
                "score": pytest.approx(0.6, abs=0.1)
            },
        ]

        # When: Querying phrase grams
        result = detector.query(queries)

        # Then: Verify query results
        assert len(result) == len(
            expected), f"Expected {len(expected)} results, got {len(result)}"
        for res, exp in zip(result, expected):
            assert res["query"] == exp["query"], f"Expected query {exp['query']}, got {res['query']}"
            assert res["phrase"] == exp["phrase"], f"Expected phrase {exp['phrase']}, got {res['phrase']}"
            assert res["score"] == exp["score"], f"Expected score {exp['score']}, got {res['score']}"


if __name__ == "__main__":
    pytest.main(["-v"])
