from typing import List, Tuple
from Levenshtein import ratio
from jet.logger import logger
from nltk.corpus import wordnet as wn
import difflib


class TextComparator:
    def __init__(self, text1: str, text2: str):
        self.text1 = text1
        self.text2 = text2
        self.words1 = text1.split()
        self.words2 = text2.split()

    def _get_color(self, similarity: float) -> str:
        """Map similarity score to ANSI color code (red to yellow to green)."""
        if similarity < 0.3:  # Adjusted for stricter red threshold
            return "\033[31m"
        elif similarity < 0.8:  # Adjusted for yellow range
            return "\033[33m"
        else:
            return "\033[32m"

    def _get_semantic_similarity(self, word1: str, word2: str) -> float:
        """Compute similarity using WordNet and Levenshtein ratio."""
        if not word1 or not word2:
            return 0.0
        # Compute lexical similarity
        lexical_sim = ratio(word1, word2)
        synsets1 = wn.synsets(word1)
        synsets2 = wn.synsets(word2)
        if not synsets1 or not synsets2:
            return lexical_sim
        max_sim = 0.0
        for syn1 in synsets1:
            for syn2 in synsets2:
                sim = syn1.wup_similarity(syn2)
                if sim is not None and sim > max_sim:
                    max_sim = sim
        # Scale WordNet similarity
        semantic_sim = max_sim * 0.5 if max_sim > 0 else 0.0
        # Use semantic similarity only for near-synonyms
        if max_sim > 0.9:  # Stricter threshold for synonyms
            return max(semantic_sim, lexical_sim)
        return lexical_sim

    def compare_texts(self) -> Tuple[str, str]:
        """Compare two texts and return color-coded versions."""
        matcher = difflib.SequenceMatcher(None, self.words1, self.words2)
        colored_text1, colored_text2 = [], []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for word1, word2 in zip(self.words1[i1:i2], self.words2[j1:j2]):
                    color = self._get_color(1.0)
                    colored_text1.append(f"{color}{word1}\033[0m")
                    colored_text2.append(f"{color}{word2}\033[0m")
            elif tag in ("replace", "delete", "insert"):
                words1_slice = self.words1[i1:i2] if tag != "insert" else []
                words2_slice = self.words2[j1:j2] if tag != "delete" else []
                max_len = max(len(words1_slice), len(words2_slice))
                words1_slice += [""] * (max_len - len(words1_slice))
                words2_slice += [""] * (max_len - len(words2_slice))
                for word1, word2 in zip(words1_slice, words2_slice):
                    similarity = self._get_semantic_similarity(word1, word2)
                    logger.debug(
                        f"Comparing '{word1}' and '{word2}', similarity: {similarity}")
                    color = self._get_color(similarity)
                    colored_text1.append(f"{color}{word1 or '-'}\033[0m")
                    colored_text2.append(f"{color}{word2 or '-'}\033[0m")
        return " ".join(colored_text1), " ".join(colored_text2)


def display_text_comparison(text1: str, text2: str) -> None:
    """Display color-coded comparison of two texts."""
    comparator = TextComparator(text1, text2)
    colored_text1, colored_text2 = comparator.compare_texts()
    print("Text 1:", colored_text1)
    print("Text 2:", colored_text2)
