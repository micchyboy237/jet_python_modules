import pytest
import random
from jet.llm.utils.mmr_diversity import sort_by_mmr_diversity


class TestSortByMMRDiversity:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.original_uniform = random.uniform
        self.score_index = 0

        def mock_uniform(a, b):
            scores = [0.8 - 0.1 * i for i in range(100)]
            result = scores[self.score_index % len(scores)]
            self.score_index += 1
            return result
        random.uniform = mock_uniform

    @pytest.fixture(autouse=True)
    def teardown(self):
        yield
        random.uniform = self.original_uniform

    def test_selects_correct_number_of_results(self):
        candidates = [
            "A sci-fi adventure about a rogue AI taking over a spaceship.",
            "A romantic comedy set in a bustling New York bakery.",
            "A thriller about a detective solving a serial killer case.",
            "A fantasy epic with dragons and ancient prophecies.",
            "A drama about a family reuniting after a tragedy."
        ]
        num_results = 3
        expected = [
            {"id": "doc_0", "rank": 1, "doc_index": 0, "score": 0.8,
             "text": "A sci-fi adventure about a rogue AI taking over a spaceship."},
            {"id": "doc_1", "rank": 2, "doc_index": 1, "score": 0.7,
             "text": "A romantic comedy set in a bustling New York bakery."},
            {"id": "doc_2", "rank": 3, "doc_index": 2, "score": 0.6,
             "text": "A thriller about a detective solving a serial killer case."},
        ]
        result = sort_by_mmr_diversity(
            candidates, num_results=num_results, lambda_param=0.5, text_diversity_weight=0.0)
        assert len(result) == len(
            expected), f"Expected {len(expected)} results, got {len(result)}"
        for r, e in zip(result, expected):
            assert r["id"] == e["id"], f"Expected id {e['id']}, got {r['id']}"
            assert r["rank"] == e["rank"], f"Expected rank {e['rank']}, got {r['rank']}"
            assert r["doc_index"] == e[
                "doc_index"], f"Expected doc_index {e['doc_index']}, got {r['doc_index']}"
            assert r["score"] == pytest.approx(
                e["score"]), f"Expected score {e['score']}, got {r['score']}"
            assert r["text"] == e["text"], f"Expected text {e['text']}, got {r['text']}"

    def test_handles_empty_candidates(self):
        candidates = []
        num_results = 3
        expected = []
        result = sort_by_mmr_diversity(candidates, num_results=num_results)
        assert result == expected, f"Expected empty list, got {result}"

    def test_applies_text_diversity_penalty(self):
        candidates = [
            "A sci-fi adventure about a rogue AI taking over a spaceship.",
            "A sci-fi adventure about a rogue AI taking over a spaceship.",  # Duplicate
            "A romantic comedy set in a bustling New York bakery."
        ]
        num_results = 2
        expected = [
            {"id": "doc_0", "rank": 1, "doc_index": 0, "score": 0.8,
             "text": "A sci-fi adventure about a rogue AI taking over a spaceship."},
            {"id": "doc_2", "rank": 2, "doc_index": 2, "score": 0.6,
             "text": "A romantic comedy set in a bustling New York bakery."},
        ]
        result = sort_by_mmr_diversity(
            candidates, num_results=num_results, lambda_param=0.5, text_diversity_weight=0.4)
        assert len(result) == len(
            expected), f"Expected {len(expected)} results, got {len(result)}"
        for r, e in zip(result, expected):
            assert r["id"] == e["id"], f"Expected id {e['id']}, got {r['id']}"
            assert r["rank"] == e["rank"], f"Expected rank {e['rank']}, got {r['rank']}"
            assert r["doc_index"] == e[
                "doc_index"], f"Expected doc_index {e['doc_index']}, got {r['doc_index']}"
            assert r["score"] == pytest.approx(
                e["score"]), f"Expected score {e['score']}, got {r['score']}"
            assert r["text"] == e["text"], f"Expected text {e['text']}, got {r['text']}"

    def test_assigns_correct_ranks(self):
        candidates = [
            "A sci-fi adventure about a rogue AI taking over a spaceship.",
            "A romantic comedy set in a bustling New York bakery.",
            "A thriller about a detective solving a serial killer case."
        ]
        num_results = 3
        expected_ranks = [1, 2, 3]
        result = sort_by_mmr_diversity(candidates, num_results=num_results)
        result_ranks = [r["rank"] for r in result]
        assert result_ranks == expected_ranks, f"Expected ranks {expected_ranks}, got {result_ranks}"

    def test_handles_num_results_greater_than_candidates(self):
        candidates = [
            "A sci-fi adventure about a rogue AI taking over a spaceship.",
            "A romantic comedy set in a bustling New York bakery."
        ]
        num_results = 5
        expected = [
            {"id": "doc_0", "rank": 1, "doc_index": 0, "score": 0.8,
             "text": "A sci-fi adventure about a rogue AI taking over a spaceship."},
            {"id": "doc_1", "rank": 2, "doc_index": 1, "score": 0.7,
             "text": "A romantic comedy set in a bustling New York bakery."},
        ]
        result = sort_by_mmr_diversity(candidates, num_results=num_results)
        assert len(result) == len(
            expected), f"Expected {len(expected)} results, got {len(result)}"
        for r, e in zip(result, expected):
            assert r["id"] == e["id"], f"Expected id {e['id']}, got {r['id']}"
            assert r["rank"] == e["rank"], f"Expected rank {e['rank']}, got {r['rank']}"
            assert r["doc_index"] == e[
                "doc_index"], f"Expected doc_index {e['doc_index']}, got {r['doc_index']}"
            assert r["score"] == pytest.approx(
                e["score"]), f"Expected score {e['score']}, got {r['score']}"
            assert r["text"] == e["text"], f"Expected text {e['text']}, got {r['text']}"
