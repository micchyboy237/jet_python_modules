from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer, util
from jet.logger import logger
from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry


class SentenceComparator:
    def __init__(self, base_sentences: List[str], sentences_to_compare: List[str]):
        # Filter out empty or whitespace-only strings
        self.base_sentences = [s for s in base_sentences if s.strip()]
        self.sentences_to_compare = [
            s for s in sentences_to_compare if s.strip()]
        logger.debug(
            f"Filtered base_sentences: {self.base_sentences}, sentences_to_compare: {self.sentences_to_compare}")
        # Raise ValueError if either list is empty after filtering
        if not self.base_sentences:
            logger.error("base_sentences is empty after filtering")
            raise ValueError("base_sentences cannot be empty after filtering")
        if not self.sentences_to_compare:
            logger.error("sentences_to_compare is empty after filtering")
            raise ValueError(
                "sentences_to_compare cannot be empty after filtering")
        # Pad lists to equal length with "-"
        if len(self.base_sentences) != len(self.sentences_to_compare):
            logger.debug(
                f"Padding lists: len(base_sentences)={len(self.base_sentences)}, len(sentences_to_compare)={len(self.sentences_to_compare)}")
            max_len = max(len(self.base_sentences),
                          len(self.sentences_to_compare))
            self.base_sentences.extend(
                ["-"] * (max_len - len(self.base_sentences)))
            self.sentences_to_compare.extend(
                ["-"] * (max_len - len(self.sentences_to_compare)))
            logger.debug(
                f"After padding: base_sentences={self.base_sentences}, sentences_to_compare={self.sentences_to_compare}")
        self.model = SentenceTransformerRegistry.load_model('all-MiniLM-L6-v2')

    def _get_color(self, similarity: float) -> str:
        """Map similarity score to ANSI color code (red to yellow to green)."""
        logger.debug(f"Selecting color for similarity: {similarity}")
        if similarity < 0.3:
            logger.debug("Returning red color \033[31m()")
            return "\033[31m"
        elif similarity < 0.95:
            logger.debug("Returning yellow color \033[33m()")
            return "\033[33m"
        else:
            logger.debug("Returning green color \033[32m()")
            return "\033[32m"

    def compare_sentences(self) -> List[Dict[str, str | float]]:
        """Compare lists of sentences and return results with similarity scores."""
        results = []
        for t1, t2 in zip(self.base_sentences, self.sentences_to_compare):
            if not t1 or not t2 or t1 == "-" or t2 == "-":
                color = self._get_color(0.0)
                results.append({
                    "sentence1": t1 or "-",
                    "sentence2": t2 or "-",
                    "similarity": 0.0,
                    "colored_sentence1": f"{color}{t1 or '-'}\033[0m",
                    "colored_sentence2": f"{color}{t2 or '-'}\033[0m"
                })
                continue

            embeddings = self.model.encode([t1, t2], convert_to_tensor=True)
            similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
            logger.debug(
                f"Comparing '{t1}' and '{t2}', similarity: {similarity}")
            color = self._get_color(similarity)
            results.append({
                "sentence1": t1,
                "sentence2": t2,
                "similarity": similarity,
                "colored_sentence1": f"{color}{t1}\033[0m",
                "colored_sentence2": f"{color}{t2}\033[0m"
            })
        return results


def display_sentence_comparison(base_sentences: List[str], sentences_to_compare: List[str]) -> None:
    """Display color-coded comparison of sentence lists."""
    comparator = SentenceComparator(base_sentences, sentences_to_compare)
    results = comparator.compare_sentences()
    for idx, result in enumerate(results, 1):
        print(f"\nPair {idx}:")
        print(f"Sentence 1: {result['colored_sentence1']}")
        print(f"Sentence 2: {result['colored_sentence2']}")
        print(f"Similarity: {result['similarity']:.4f}")
