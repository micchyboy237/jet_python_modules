from typing import List, Dict, Any, Set, Tuple, TypedDict
from spellchecker import SpellChecker
from sentence_transformers import SentenceTransformer, util
import numpy as np
import logging

from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry
from jet.models.model_types import EmbedModelType

# Set up debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class SearchResult(TypedDict):
    rank: int
    score: float
    text: str


class SpellCorrectedSearchEngine:
    def __init__(self, model_name: EmbedModelType = "all-MiniLM-L6-v2"):
        """Initialize spell checker and sentence transformer model."""
        self.spell_checker = SpellChecker()
        self.model = SentenceTransformerRegistry.load_model(model_name)
        self.documents: List[Dict[str, Any]] = []
        self.corrected_documents: List[Dict[str, Any]] = []
        self.corrections: List[List[Dict[str, str]]] = []
        self.document_words: Set[str] = set()
        self.query_words: Set[str] = set()
        self.embeddings: np.ndarray = None

    def collect_unique_words(self, documents: List[Dict[str, Any]], queries: List[str]) -> None:
        """Collect unique words from documents and queries separately."""
        for doc in documents:
            words = set(doc["content"].lower().split())
            self.document_words.update(words)
        for query in queries:
            words = set(query.lower().split())
            self.query_words.update(words)
        known_words = set(self.spell_checker.known(self.document_words))
        self.document_words = known_words | self.query_words
        logger.debug(f"Document words: {self.document_words}")
        logger.debug(f"Query words: {self.query_words}")
        self.spell_checker.word_frequency.load_words(
            self.document_words | self.query_words)

    def correct_text(self, text: str) -> Tuple[str, List[Dict[str, str]]]:
        """Correct misspellings in a given text and track corrections."""
        words = text.split()
        corrected_words = []
        corrections = []
        for word in words:
            lower_word = word.lower()
            corrected = self.spell_checker.correction(lower_word)
            if corrected is None:
                corrected = word
                logger.debug(f"No correction for '{word}'")
            else:
                corrected = corrected if word.islower() else corrected.capitalize()
                logger.debug(f"Correcting '{word}' to '{corrected}'")
            corrected_words.append(corrected)
            if corrected.lower() != lower_word:
                corrections.append({"original": word, "corrected": corrected})
        return " ".join(corrected_words), corrections

    def add_documents(self, documents: List[Dict[str, Any]], queries: List[str] = None) -> None:
        """Add documents, correct misspellings, track corrections, and compute embeddings."""
        self.documents = documents
        self.collect_unique_words(documents, queries or [])
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

    def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        """Perform semantic search on corrected documents and rank results."""
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.embeddings)[0]
        top_indices = np.argsort(cos_scores.cpu().numpy())[::-1][:limit]

        results: List[SearchResult] = []
        for rank, idx in enumerate(top_indices, 1):
            cos_score = cos_scores[idx].item()
            doc = self.corrected_documents[idx]
            num_corrections = len(self.corrections[idx])
            num_words = len(doc["content"].split())
            correction_penalty = num_corrections / num_words if num_words > 0 else 0
            # Reduced penalty impact
            final_score = cos_score * (1 - correction_penalty * 0.5)
            logger.debug(
                f"Document {doc['id']}: Cosine score={cos_score:.4f}, Penalty={correction_penalty:.4f}, Final score={final_score:.4f}")
            results.append({
                "rank": rank,
                "score": final_score,
                "text": doc["content"]
            })
        return results
