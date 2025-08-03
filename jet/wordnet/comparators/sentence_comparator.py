from typing import Tuple
from sentence_transformers import SentenceTransformer, util
from jet.logger import logger
from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry


class SentenceComparator:
    def __init__(self, text1: str, text2: str):
        self.text1 = text1
        self.text2 = text2
        self.model = SentenceTransformerRegistry.load_model('all-MiniLM-L6-v2')

    def _get_color(self, similarity: float) -> str:
        """Map similarity score to ANSI color code (red to yellow to green)."""
        logger.debug(f"Selecting color for similarity: {similarity}")
        if similarity < 0.3:
            logger.debug("Returning red color (\033[31m)")
            return "\033[31m"
        elif similarity < 0.95:  # Adjusted threshold
            logger.debug("Returning yellow color (\033[33m)")
            return "\033[33m"
        else:
            logger.debug("Returning green color (\033[32m)")
            return "\033[32m"

    def compare_sentences(self) -> Tuple[str, str]:
        """Compare two sentences using sentence transformers and return color-coded versions."""
        if not self.text1 or not self.text2:
            color = self._get_color(0.0)
            return (
                f"{color}{self.text1 or '-'}\033[0m",
                f"{color}{self.text2 or '-'}\033[0m"
            )

        # Compute sentence embeddings
        embeddings = self.model.encode(
            [self.text1, self.text2], convert_to_tensor=True)
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        logger.debug(
            f"Comparing '{self.text1}' and '{self.text2}', similarity: {similarity}")

        color = self._get_color(similarity)
        return (
            f"{color}{self.text1}\033[0m",
            f"{color}{self.text2}\033[0m"
        )


def display_sentence_comparison(text1: str, text2: str) -> None:
    """Display color-coded comparison of two sentences."""
    comparator = SentenceComparator(text1, text2)
    colored_text1, colored_text2 = comparator.compare_sentences()
    print("Sentence 1:", colored_text1)
    print("Sentence 2:", colored_text2)
