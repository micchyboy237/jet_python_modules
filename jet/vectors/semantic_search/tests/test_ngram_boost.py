import pytest
import numpy as np
from jet.vectors.semantic_search.booster import boost_ngram_score
from jet.vectors.semantic_search.search_types import Match


class TestNgramScoreBoost:
    def test_longer_ngram_gets_higher_boost(self):
        """Test that a longer n-gram match results in a higher score boost."""
        # Given matches with different lengths
        short_match = [Match(text="react", start_idx=0, end_idx=5)]
        long_match = [Match(text="react native", start_idx=0, end_idx=12)]
        base_score = 0.5

        # When boosting scores
        expected_short = 0.5 * (1 + 1.5 * (np.log1p(5) / np.log1p(100)))
        expected_long = 0.5 * (1 + 1.5 * (np.log1p(12) / np.log1p(100)))
        result_short = boost_ngram_score(short_match, base_score)
        result_long = boost_ngram_score(long_match, base_score)

        # Then the longer match has a higher boost
        assert result_long > result_short, "Longer n-gram should have higher boost"
        assert pytest.approx(result_short, 0.01) == expected_short
        assert pytest.approx(result_long, 0.01) == expected_long

    def test_no_matches_returns_base_score(self):
        """Test that no matches returns the original base score."""
        # Given no matches
        matches = []
        base_score = 0.7

        # When boosting the score
        expected = 0.7
        result = boost_ngram_score(matches, base_score)

        # Then the base score is returned
        assert result == expected, "No matches should return base score"

    def test_multiple_matches_uses_longest(self):
        """Test that the longest match is used for boosting when multiple matches exist."""
        # Given multiple matches with different lengths
        matches = [
            Match(text="react", start_idx=0, end_idx=5),
            Match(text="react native", start_idx=0, end_idx=12),
            Match(text="react native dev", start_idx=0, end_idx=16)
        ]
        base_score = 0.6

        # When boosting the score
        expected = 0.6 * (1 + 1.5 * (np.log1p(16) / np.log1p(100)))
        result = boost_ngram_score(matches, base_score)

        # Then the longest match is used for the boost
        assert pytest.approx(
            result, 0.01) == expected, "Should use longest match for boost"
