"""Ensemble methods for combining multiple search signals."""

from typing import Callable

import numpy as np
from jet.adapters.llama_cpp.embed_utils import embed
from jet.adapters.llama_cpp.rerank_utils import rerank
from rank_bm25 import BM25Okapi


class EnsembleSearch:
    """
    Ensemble search combining multiple ranking signals with configurable weights.
    """

    def __init__(self, weights: dict[str, float] = None):
        """
        Initialize ensemble with signal weights.

        Args:
            weights: Dict mapping signal names to weights.
                    Default: {"embedding": 0.4, "keyword": 0.3, "reranker": 0.3}
        """
        self.weights = weights or {
            "embedding": 0.4,
            "keyword": 0.3,
            "reranker": 0.3,
        }
        self._validate_weights()

    def _validate_weights(self):
        """Validate that weights sum to approximately 1.0."""
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            print(f"Warning: Weights sum to {total}, normalizing...")
            self.weights = {k: v / total for k, v in self.weights.items()}

    def search(
        self,
        query: str,
        documents: list[str],
        top_k: int = 10,
        return_details: bool = False,
    ) -> list[dict]:
        """
        Ensemble search combining multiple signals.

        Args:
            query: Search query
            documents: List of documents to search
            top_k: Number of results to return
            return_details: If True, return individual signal scores

        Returns:
            Ranked results with ensemble scores
        """
        signals = {}

        # 1. Embedding signal
        if "embedding" in self.weights and self.weights["embedding"] > 0:
            signals["embedding"] = self._get_embedding_scores(query, documents)

        # 2. Keyword signal (BM25)
        if "keyword" in self.weights and self.weights["keyword"] > 0:
            signals["keyword"] = self._get_keyword_scores(query, documents)

        # 3. Reranker signal
        if "reranker" in self.weights and self.weights["reranker"] > 0:
            signals["reranker"] = self._get_reranker_scores(query, documents)

        # Combine signals
        ensemble_scores = np.zeros(len(documents))
        for signal_name, scores in signals.items():
            ensemble_scores += self.weights[signal_name] * scores

        # Get top results
        top_indices = np.argsort(ensemble_scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            result = {
                "document": documents[idx],
                "score": float(ensemble_scores[idx]),
                "index": idx,
            }

            if return_details:
                result["signals"] = {
                    name: float(scores[idx]) for name, scores in signals.items()
                }

            results.append(result)

        return results

    def _get_embedding_scores(self, query: str, documents: list[str]) -> np.ndarray:
        """Get normalized embedding similarity scores."""
        query_emb = embed(query)
        doc_embs = embed(documents)

        scores = np.array(
            [_cosine_similarity(query_emb, doc_emb) for doc_emb in doc_embs]
        )

        # Normalize to 0-1
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())

        return scores

    def _get_keyword_scores(self, query: str, documents: list[str]) -> np.ndarray:
        """Get normalized BM25 scores."""
        tokenized_docs = [doc.lower().split() for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        # Normalize to 0-1
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())

        return scores

    def _get_reranker_scores(self, query: str, documents: list[str]) -> np.ndarray:
        """Get normalized reranker scores."""
        results = rerank(query, documents, top_n=len(documents))

        # Create score array
        scores = np.zeros(len(documents))
        for r in results:
            scores[r["index"]] = r["score"]

        # Normalize to 0-1 (reranker scores can be negative)
        scores = 1 / (1 + np.exp(-scores))  # Sigmoid normalization

        return scores

    def add_signal(
        self,
        name: str,
        weight: float,
        score_function: Callable[[str, list[str]], np.ndarray],
    ):
        """
        Add a custom signal to the ensemble.

        Args:
            name: Signal name
            weight: Weight for this signal
            score_function: Function that takes (query, documents) and returns scores
        """
        self.weights[name] = weight
        self._validate_weights()
        self.custom_signals[name] = score_function


def compare_weights(
    query: str,
    documents: list[str],
    weight_configs: list[dict],
    top_k: int = 5,
):
    """
    Compare different weight configurations for A/B testing.

    Args:
        query: Search query
        documents: List of documents
        weight_configs: List of weight dictionaries to compare
        top_k: Number of results to show per config
    """
    for i, weights in enumerate(weight_configs):
        print(f"\nConfiguration {i + 1}: {weights}")
        print("-" * 40)

        ensemble = EnsembleSearch(weights=weights)
        results = ensemble.search(query, documents, top_k=top_k, return_details=True)

        for j, r in enumerate(results, 1):
            signal_str = " ".join(
                [f"{name}:{score:.3f}" for name, score in r.get("signals", {}).items()]
            )
            print(f"{j}. [{r['score']:.4f}] {r['document'][:80]}...")
            print(f"   Signals: {signal_str}")


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


if __name__ == "__main__":
    query = "What is a giant panda?"
    documents = [
        "The giant panda is a bear species endemic to China.",
        "Python is a high-level programming language.",
        "Bears are carnivoran mammals of the family Ursidae.",
        "Machine learning is a subset of artificial intelligence.",
        "Pandas eat bamboo and live in mountainous regions.",
        "The panda's diet is 99% bamboo despite being classified as carnivores.",
        "Data science combines statistics, computer science, and domain expertise.",
        "China has established over 50 panda reserves in Sichuan province.",
        "JavaScript is commonly used for web development.",
        "Conservation efforts have helped increase wild panda populations.",
    ]

    print("=" * 60)
    print("ENSEMBLE SEARCH (Default Weights)")
    print("=" * 60)
    print(f"Query: {query}\n")

    ensemble = EnsembleSearch()
    results = ensemble.search(query, documents, top_k=5, return_details=True)

    for i, r in enumerate(results, 1):
        print(f"{i}. [ensemble:{r['score']:.4f}] {r['document']}")
        signal_str = " | ".join(
            [f"{name}:{score:.3f}" for name, score in r["signals"].items()]
        )
        print(f"   {signal_str}")

    # A/B Testing
    print("\n" + "=" * 60)
    print("A/B TESTING DIFFERENT WEIGHT CONFIGURATIONS")
    print("=" * 60)

    weight_configs = [
        {"embedding": 0.4, "keyword": 0.3, "reranker": 0.3},  # Balanced
        {"embedding": 0.7, "keyword": 0.3, "reranker": 0.0},  # Embedding-heavy
        {"embedding": 0.2, "keyword": 0.2, "reranker": 0.6},  # Reranker-heavy
        {"embedding": 0.3, "keyword": 0.6, "reranker": 0.1},  # Keyword-heavy
    ]

    compare_weights(query, documents, weight_configs, top_k=3)
