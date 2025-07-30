from spellchecker import SpellChecker
from typing import List, TypedDict, Optional


class SearchResult(TypedDict):
    rank: int
    score: float
    text: str


class KeywordVectorSearchSpellChecker:
    def __init__(self, language: str = "en"):
        self.spell_checker = SpellChecker(language=language)
        self.words: List[str] = []

    def build_index(self, words: Optional[List[str]] = None):
        """Build word frequency list for spell checking. Uses default dictionary if no words provided."""
        if words:
            self.words = words
            # Boost frequency for custom words to prioritize them
            for word in words:
                # Higher frequency for custom words
                self.spell_checker.word_frequency.add(word, val=1000)
        else:
            self.words = list(self.spell_checker.word_frequency.words())

    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        """Search for the k most likely corrections for the query."""

        # Check if query is in dictionary
        if query in self.spell_checker:
            score = self.spell_checker.word_usage_frequency(query)
            return [SearchResult(rank=1, score=score, text=query)]

        # Get top correction
        top_correction = self.spell_checker.correction(query)

        # Get candidates, filter by self.words if custom dictionary is used
        candidates = self.spell_checker.candidates(query) or set()
        if self.words and self.spell_checker.word_frequency.unique_words == len(self.words):
            candidates = {c for c in candidates if c in self.words}

        # Build results, prioritizing top correction
        results = []
        if top_correction and (not self.words or top_correction in self.words):
            score = self.spell_checker.word_usage_frequency(top_correction)
            results.append(SearchResult(
                rank=1, score=score, text=top_correction))

        # Add remaining candidates
        remaining_candidates = [c for c in candidates if c != top_correction]
        results.extend([
            SearchResult(
                rank=i + 2, score=self.spell_checker.word_usage_frequency(word), text=word)
            for i, word in enumerate(remaining_candidates) if i < k - 1
        ])

        # Sort by score (frequency) in descending order and reassign ranks
        sorted_results = sorted(
            results, key=lambda x: x["score"], reverse=True)[:k]
        for i, result in enumerate(sorted_results):
            result["rank"] = i + 1
        return sorted_results
