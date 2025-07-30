from spellchecker import SpellChecker
from typing import List, Union, TypedDict, Optional
import re


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
        # Split on whitespace and remove punctuation
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        return words

    def search(self, input_data: Union[str, List[str]], k: int = 5) -> List[SearchResult]:
        """Search for misspellings in input string or list of documents.

        Args:
            input_data: Single string or list of document strings to check
            k: Maximum number of corrections to return per misspelled word

        Returns:
            List of SearchResult dictionaries containing corrections
        """
        # Handle single string or list of documents
        documents = [input_data] if isinstance(input_data, str) else input_data
        all_results = []

        for doc in documents:
            if not doc:
                continue
            words = self._process_text(doc)

            for word in words:
                if word in self.spell_checker and (not self.words or word in self.words):
                    continue  # Skip correctly spelled words

                # Get correction and candidates
                top_correction = self.spell_checker.correction(word)
                candidates = self.spell_checker.candidates(word) or set()

                # Filter candidates if custom word list is provided
                if self.words:
                    candidates = {c for c in candidates if c in self.words}

                # Build results for this word
                word_results = []
                if top_correction and (not self.words or top_correction in self.words):
                    score = self.spell_checker.word_usage_frequency(
                        top_correction)
                    word_results.append(SearchResult(
                        rank=1,
                        score=score,
                        text=top_correction,
                        original=word
                    ))

                # Add remaining candidates
                remaining_candidates = [
                    c for c in candidates if c != top_correction]
                word_results.extend([
                    SearchResult(
                        rank=i + 2,
                        score=self.spell_checker.word_usage_frequency(c),
                        text=c,
                        original=word
                    ) for i, c in enumerate(remaining_candidates) if i < k - 1
                ])

                # Sort and limit results for this word
                sorted_word_results = sorted(
                    word_results,
                    key=lambda x: x["score"],
                    reverse=True
                )[:k]
                all_results.extend(sorted_word_results)

        # Sort all results and assign final ranks
        sorted_results = sorted(
            all_results, key=lambda x: x["score"], reverse=True)[:k]
        for i, result in enumerate(sorted_results):
            result["rank"] = i + 1

        return sorted_results
