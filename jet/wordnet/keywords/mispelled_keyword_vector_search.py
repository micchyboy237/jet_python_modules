from spellchecker import SpellChecker
from typing import List, Union, TypedDict, Optional
import re
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class SearchResult(TypedDict):
    rank: int
    score: float
    text: str
    original: str


class MispelledKeywordVectorSearch:
    def __init__(self, language: str = "en"):
        """Initialize spell checker with specified language."""
        self.spell_checker = SpellChecker(language=language)
        self.words: List[str] = []

    def build_index(self, words: Optional[List[str]] = None):
        """Build word frequency index for spell checking.

        Args:
            words: Optional list of words to build custom index. Uses default dictionary if None.
        """
        if words:
            self.words = words
            for word in words:
                self.spell_checker.word_frequency.add(word, val=1000)
        else:
            self.words = list(self.spell_checker.word_frequency.words())

    def _process_text(self, text: str) -> List[str]:
        """Split text into words and filter out non-alphabetic tokens."""
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        return words

    def _edit_distance(self, word1: str, word2: str) -> int:
        """Calculate Levenshtein edit distance between two words."""
        if word1 == word2:
            return 0
        dp = [[0] * (len(word2) + 1) for _ in range(len(word1) + 1)]
        for i in range(len(word1) + 1):
            dp[i][0] = i
        for j in range(len(word2) + 1):
            dp[0][j] = j
        for i in range(1, len(word1) + 1):
            for j in range(1, len(word2) + 1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
        return dp[len(word1)][len(word2)]

    def _normalize_score(self, frequency: float, edit_dist: int, max_freq: float) -> float:
        """Normalize score combining word frequency and edit distance."""
        max_dist = 5  # Reasonable max edit distance for normalization
        freq_weight = 0.7
        dist_weight = 0.3
        norm_freq = frequency / max_freq if max_freq > 0 else frequency
        norm_dist = 1 - (min(edit_dist, max_dist) /
                         max_dist) if edit_dist > 0 else 1
        score = freq_weight * norm_freq + dist_weight * norm_dist
        logger.debug(
            f"Normalizing score: freq={frequency}, edit_dist={edit_dist}, max_freq={max_freq}, norm_freq={norm_freq}, norm_dist={norm_dist}, final_score={score}")
        return score

    def search(self, input_data: Union[str, List[str]], k: int = 5) -> List[SearchResult]:
        """Search for misspellings in input string or list of documents.

        Args:
            input_data: Single string or list of document strings to check
            k: Maximum number of corrections to return

        Returns:
            List of SearchResult dictionaries containing unique corrections
        """
        documents = [input_data] if isinstance(input_data, str) else input_data
        all_results = []
        seen = set()  # Track (text, original) pairs for deduplication
        # Calculate max frequency from known words
        max_freq = max((self.spell_checker.word_usage_frequency(word)
                       for word in self.spell_checker.word_frequency.words()), default=1.0)
        logger.debug(f"Max frequency calculated: {max_freq}")

        for doc in documents:
            if not doc:
                continue
            words = self._process_text(doc)
            logger.debug(f"Processed words: {words}")

            for word in words:
                logger.debug(f"Checking word: {word}")
                if word in self.spell_checker and (not self.words or word in self.words):
                    logger.debug(f"Word '{word}' is correctly spelled")
                    continue

                top_correction = self.spell_checker.correction(word)
                candidates = self.spell_checker.candidates(word) or set()
                logger.debug(f"Candidates for '{word}': {candidates}")
                if self.words:
                    candidates = {c for c in candidates if c in self.words}
                    logger.debug(f"Filtered candidates: {candidates}")

                word_results = []
                if top_correction and (not self.words or top_correction in self.words):
                    freq = self.spell_checker.word_usage_frequency(
                        top_correction)
                    edit_dist = self._edit_distance(word, top_correction)
                    score = self._normalize_score(freq, edit_dist, max_freq)
                    key = (top_correction, word)
                    if key not in seen:
                        seen.add(key)
                        word_results.append(SearchResult(
                            rank=1,
                            score=score,
                            text=top_correction,
                            original=word
                        ))
                        logger.debug(
                            f"Top correction for '{word}': {top_correction}, score={score}")

                remaining_candidates = [
                    c for c in candidates if c != top_correction]
                for i, candidate in enumerate(remaining_candidates):
                    if i >= k - 1:
                        break
                    key = (candidate, word)
                    if key in seen:
                        continue
                    seen.add(key)
                    freq = self.spell_checker.word_usage_frequency(candidate)
                    edit_dist = self._edit_distance(word, candidate)
                    score = self._normalize_score(freq, edit_dist, max_freq)
                    word_results.append(SearchResult(
                        rank=i + 2,
                        score=score,
                        text=candidate,
                        original=word
                    ))
                    logger.debug(
                        f"Candidate correction for '{word}': {candidate}, score={score}")

                all_results.extend(word_results)

        sorted_results = sorted(
            all_results, key=lambda x: x["score"], reverse=True)[:k]
        for i, result in enumerate(sorted_results):
            result["rank"] = i + 1
        logger.debug(f"Final results: {sorted_results}")
        return sorted_results
