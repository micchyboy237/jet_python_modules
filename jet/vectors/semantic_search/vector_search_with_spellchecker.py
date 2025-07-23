from typing import List, Dict, Any, Set, Tuple
from spellchecker import SpellChecker
from sentence_transformers import SentenceTransformer, util
import numpy as np
import logging

from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry
from jet.models.model_types import EmbedModelType

# Set up debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class SpellCorrectedSearchEngine:
    def __init__(self, model_name: EmbedModelType = "all-MiniLM-L6-v2"):
        """Initialize spell checker and sentence transformer model."""
        self.spell_checker = SpellChecker()
        self.model = SentenceTransformerRegistry.load_model(model_name)
        self.documents: List[Dict[str, Any]] = []
        self.corrected_documents: List[Dict[str, Any]] = []
        self.corrections: List[List[Dict[str, str]]] = []
        self.custom_words: Set[str] = set()

    def build_custom_dictionary(self, documents: List[Dict[str, Any]], queries: List[str]) -> None:
        """Build a custom dictionary from document and query words, excluding likely misspellings."""
        known_words = set(self.spell_checker.known(
            [word.lower() for word in self.spell_checker.word_frequency.dictionary]))
        for doc in documents:
            words = doc["content"].lower().split()
            # Only add words that are likely correct or from queries
            self.custom_words.update(
                word for word in words if word in known_words)
        for query in queries:
            words = query.lower().split()
            self.custom_words.update(words)
        logger.debug(f"Custom dictionary: {self.custom_words}")
        self.spell_checker.word_frequency.load_words(self.custom_words)

    def correct_text(self, text: str) -> Tuple[str, List[Dict[str, str]]]:
        """Correct misspellings in a given text and track corrections."""
        words = text.split()
        corrected_words = []
        corrections = []
        for word in words:
            lower_word = word.lower()
            corrected = self.spell_checker.correction(lower_word)
            if corrected is None:
                corrected = word  # Fallback to original if no correction
                logger.debug(f"No correction for '{word}'")
            else:
                # Preserve original case
                corrected = corrected if word.islower() else corrected.capitalize()
                logger.debug(f"Correcting '{word}' to '{corrected}'")
            corrected_words.append(corrected)
            if corrected.lower() != lower_word:
                corrections.append({"original": word, "corrected": corrected})
        return " ".join(corrected_words), corrections

    def add_documents(self, documents: List[Dict[str, Any]], queries: List[str] = None) -> None:
        """Add documents, correct misspellings, track corrections, and compute embeddings."""
        self.documents = documents
        self.build_custom_dictionary(documents, queries or [])
        self.corrected_documents = []
        self.corrections = []
        for doc in documents:
            corrected_content, doc_corrections = self.correct_text(
                doc["content"])
            self.corrected_documents.append(
                {"id": doc["id"], "content": corrected_content})
            self.corrections.append(doc_corrections)
            logger.debug(
                f"Document {doc['id']} corrections: {doc_corrections}")
        texts = [doc["content"] for doc in self.corrected_documents]
        self.embeddings = self.model.encode(texts, convert_to_tensor=True)

    def get_corrections(self, doc_id: int) -> List[Dict[str, str]]:
        """Return the list of corrections for a specific document."""
        for i, doc in enumerate(self.documents):
            if doc["id"] == doc_id:
                return self.corrections[i]
        return []

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Perform semantic search on corrected documents."""
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.embeddings)[0]
        top_indices = np.argsort(cos_scores.cpu().numpy())[::-1][:limit]

        results = []
        for idx in top_indices:
            score = cos_scores[idx].item()
            doc = self.corrected_documents[idx]
            results.append(
                {"id": doc["id"], "content": doc["content"], "score": score})
        return results
