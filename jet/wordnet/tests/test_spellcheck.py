from typing import List, Set
import pytest
from rapidfuzz import fuzz
from jet.wordnet.spellcheck import correct_typos
from jet.logger import logger


class TestCorrectTypos:
    def test_correct_single_typo(self):
        query_tokens = ["instal"]
        all_tokens = {"install", "configure", "package"}
        expected_tokens = ["install"]
        result = correct_typos(query_tokens, all_tokens, threshold=60.0)
        logger.debug("Corrected tokens for query %s: %s, expected: %s",
                     query_tokens, result, expected_tokens)
        assert result == expected_tokens, f"Expected tokens {expected_tokens}, got {result}"

    def test_no_correction_needed(self):
        query_tokens = ["install", "package"]
        all_tokens = {"install", "configure", "package"}
        expected_tokens = ["install", "package"]
        result = correct_typos(query_tokens, all_tokens, threshold=60.0)
        logger.debug("Corrected tokens for query %s: %s, expected: %s",
                     query_tokens, result, expected_tokens)
        assert result == expected_tokens, f"Expected tokens {expected_tokens}, got {result}"

    def test_non_matching_term(self):
        query_tokens = ["xyz"]
        all_tokens = {"install", "configure", "package"}
        expected_tokens = ["xyz"]
        result = correct_typos(query_tokens, all_tokens, threshold=60.0)
        logger.debug("Corrected tokens for query %s: %s, expected: %s",
                     query_tokens, result, expected_tokens)
        assert result == expected_tokens, f"Expected tokens {expected_tokens}, got {result}"

    def test_empty_query_tokens(self):
        query_tokens: List[str] = []
        all_tokens = {"install", "configure", "package"}
        expected_tokens: List[str] = []
        result = correct_typos(query_tokens, all_tokens, threshold=60.0)
        logger.debug("Corrected tokens for empty query: %s, expected: %s",
                     result, expected_tokens)
        assert result == expected_tokens, f"Expected tokens {expected_tokens}, got {result}"

    def test_custom_threshold(self):
        query_tokens = ["instal"]
        all_tokens = {"install", "configure", "package"}
        expected_tokens = ["instal"]  # No correction with high threshold
        result = correct_typos(query_tokens, all_tokens,
                               threshold=95.0)  # Increased from 90.0
        logger.debug("Corrected tokens for query %s: %s, expected: %s",
                     query_tokens, result, expected_tokens)
        assert result == expected_tokens, f"Expected tokens {expected_tokens}, got {result}"

    def test_case_insensitive(self):
        query_tokens = ["INSTAL"]
        all_tokens = {"install", "configure", "package"}
        expected_tokens = ["install"]
        result = correct_typos(query_tokens, all_tokens, case_sensitive=False)
        logger.debug("Corrected tokens for query %s: %s, expected: %s",
                     query_tokens, result, expected_tokens)
        assert result == expected_tokens, f"Expected tokens {expected_tokens}, got {result}"

    def test_custom_scorer(self):
        query_tokens = ["instal"]
        all_tokens = {"install", "configure", "package"}
        expected_tokens = ["install"]
        result = correct_typos(query_tokens, all_tokens, scorer=fuzz.ratio)
        logger.debug("Corrected tokens for query %s: %s, expected: %s",
                     query_tokens, result, expected_tokens)
        assert result == expected_tokens, f"Expected tokens {expected_tokens}, got {result}"

    def test_max_corrections(self):
        query_tokens = ["instal", "configur"]
        all_tokens = {"install", "configure", "package"}
        expected_tokens = ["install", "configur"]  # Only first term corrected
        result = correct_typos(query_tokens, all_tokens, max_corrections=1)
        logger.debug("Corrected tokens for query %s: %s, expected: %s",
                     query_tokens, result, expected_tokens)
        assert result == expected_tokens, f"Expected tokens {expected_tokens}, got {result}"

    def test_ignore_tokens(self):
        query_tokens = ["instal", "config"]
        all_tokens = {"install", "configure", "package"}
        # "config" not corrected due to ignore
        expected_tokens = ["install", "config"]
        result = correct_typos(query_tokens, all_tokens,
                               ignore_tokens={"config"})
        logger.debug("Corrected tokens for query %s: %s, expected: %s",
                     query_tokens, result, expected_tokens)
        assert result == expected_tokens, f"Expected tokens {expected_tokens}, got {result}"

    def test_return_details(self):
        query_tokens = ["instal"]
        all_tokens = {"install", "configure", "package"}
        expected_details = [
            ("instal", "install", pytest.approx(92.3076923076923, abs=5.0))]
        result = correct_typos(query_tokens, all_tokens, return_details=True)
        logger.debug("Corrected details for query %s: %s, expected: %s",
                     query_tokens, result, expected_details)
        assert len(result) == len(
            expected_details), f"Expected {len(expected_details)} results, got {len(result)}"
        for (orig, corr, score), (exp_orig, exp_corr, exp_score) in zip(result, expected_details):
            assert orig == exp_orig, f"Expected original {exp_orig}, got {orig}"
            assert corr == exp_corr, f"Expected corrected {exp_corr}, got {corr}"
            assert score == exp_score, f"Expected score {exp_score}, got {score}"
