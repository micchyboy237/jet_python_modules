import math
from dataclasses import dataclass
from typing import Any, TypedDict

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)

from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding
from jet.adapters.llama_cpp.types import LLAMACPP_EMBED_KEYS, SearchResultType
from jet.adapters.llama_cpp.vector_search import VectorSearch
from jet.logger import CustomLogger
from rank_bm25 import BM25Okapi

# ──────────────────────────────────────────────────────────────────────────────
# Globals / Helpers
# ──────────────────────────────────────────────────────────────────────────────

stop_words = set(stopwords.words("english"))


def better_tokenize(text: str) -> list[str]:
    tokens = word_tokenize(text.lower())
    return [t for t in tokens if t.isalnum() and t not in stop_words]


# ──────────────────────────────────────────────────────────────────────────────
# Types (unchanged)
# ──────────────────────────────────────────────────────────────────────────────


class HybridSearchResult(TypedDict):
    rank: int
    index: int
    id: str | None
    text: str
    dense_score: float
    sparse_score: float
    hybrid_score: float
    category: str  # human-readable label
    category_level: int  # 0=Very Low ... 4=Very High (ordered)


# ──────────────────────────────────────────────────────────────────────────────
# Fusion functions
# ──────────────────────────────────────────────────────────────────────────────


def reciprocal_rank_fusion(
    dense_results: list[SearchResultType],
    sparse_results: list[SearchResultType],
    k: float = 60.0,
    limit: int | None = None,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> list[HybridSearchResult]:
    """
    Weighted Reciprocal Rank Fusion.

    Allows giving more importance to dense or sparse rankings.
    """
    scores: dict[int, float] = {}
    ranks_dense: dict[int, int] = {
        r.get("index"): i + 1 for i, r in enumerate(dense_results)
    }
    ranks_sparse: dict[int, int] = {
        r["index"]: i + 1 for i, r in enumerate(sparse_results)
    }

    all_indices = set(ranks_dense) | set(ranks_sparse)

    for idx in all_indices:
        r_d = ranks_dense.get(idx, float("inf"))
        r_s = ranks_sparse.get(idx, float("inf"))
        contrib_dense = dense_weight / (k + r_d)
        contrib_sparse = sparse_weight / (k + r_s)
        scores[idx] = contrib_dense + contrib_sparse

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if limit is not None:
        ranked = ranked[:limit]

    final = []
    for pos, (idx, hybrid_score) in enumerate(ranked, 1):
        ref = next((r for r in dense_results if r.get("index") == idx), None) or next(
            (r for r in sparse_results if r["index"] == idx), None
        )
        if not ref:
            continue

        cat, cat_level = get_relevance_category(round(hybrid_score, 6))
        final.append(
            {
                "rank": pos,
                "index": idx,
                "id": ref.get("id"),
                "text": ref["text"],
                "dense_score": ref["score"],
                "sparse_score": next(
                    (r["score"] for r in sparse_results if r["index"] == idx), 0.0
                ),
                "hybrid_score": round(hybrid_score, 6),
                "category": cat,
                "category_level": cat_level,
            }
        )

    return final


def get_relevance_category(score: float) -> tuple[str, int]:
    """
    Returns (label: str, level: int) based on hybrid score.

    Higher level = better relevance.
    Thresholds tuned for typical RRF scores with k≈10–15 and weights ≈1.5/0.7.
    """
    if score >= 0.135:
        return "Very High", 4
    elif score >= 0.110:
        return "High", 3
    elif score >= 0.080:
        return "Medium", 2
    elif score >= 0.040:
        return "Low", 1
    else:
        return "Very Low", 0


@dataclass
class HybridSearch:
    """
    Hybrid (dense + sparse) retriever using Weighted Reciprocal Rank Fusion by default.
    """

    documents: list[str]
    ids: list[str] | None = None
    embedding_model: LlamacppEmbedding | str = "nomic-embed-text"
    bm25: BM25Okapi | None = None
    vector_search: VectorSearch | None = None
    fusion_k: float = 60.0  # base value — will be adapted in search()
    dense_weight: float = 1.5  # stronger semantic preference — fixes salad query
    sparse_weight: float = 0.7
    min_hybrid_score: float = 0.015
    logger: CustomLogger | None = None

    def __post_init__(self):
        if self.logger is None:
            self.logger = CustomLogger()

        # Build sparse index with improved tokenization
        tokenized_docs = [better_tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)

        # Initialize vector search (dense retriever)
        if isinstance(self.embedding_model, str):
            self.vector_search = VectorSearch(
                model=self.embedding_model,
                normalize=True,
                query_prefix="search_query: ",
                document_prefix="search_document: ",
            )
        else:
            self.vector_search = VectorSearch(model=self.embedding_model)

    @classmethod
    def from_documents(
        cls,
        documents: list[str],
        ids: list[str] | None = None,
        model: LLAMACPP_EMBED_KEYS | LlamacppEmbedding = "nomic-embed-text",
        **kwargs: Any,
    ) -> "HybridSearch":
        if ids is not None and len(ids) != len(documents):
            raise ValueError("ids must be None or same length as documents")

        # Forward all extra kwargs (allows overriding weights, k, etc.)
        return cls(documents=documents, ids=ids, embedding_model=model, **kwargs)

    def _sparse_search(
        self, query: str, top_k: int | None = None
    ) -> list[SearchResultType]:
        tokenized_query = better_tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(
            [(i, score) for i, score in enumerate(scores) if score > 0],
            key=lambda x: x[1],
            reverse=True,
        )

        if top_k is not None:
            ranked = ranked[:top_k]

        results: list[SearchResultType] = []
        for idx, score in ranked:
            item: SearchResultType = {
                "index": idx,
                "text": self.documents[idx],
                "score": float(score),
            }
            if self.ids:
                item["id"] = self.ids[idx]
            results.append(item)

        return results

    def search(
        self,
        query: str,
        top_k: int = 5,
        dense_top_k: int | None = None,
        sparse_top_k: int | None = None,
        fusion_k: float | None = None,
        dense_weight: float | None = None,
        sparse_weight: float | None = None,
        min_hybrid_score: float | None = None,
        debug: bool = False,
        **search_kwargs: Any,
    ) -> list[HybridSearchResult]:
        """
        Perform hybrid search with configurable weighting and adaptive k.

        Args:
            debug: If True, print top dense/sparse candidates regardless of log level
            **search_kwargs: Allows passing min_hybrid_score, etc. for one-off calls
        """
        # ────── Parameter resolution ──────
        use_k = fusion_k if fusion_k is not None else self.fusion_k
        use_dense_w = dense_weight if dense_weight is not None else self.dense_weight
        use_sparse_w = (
            sparse_weight if sparse_weight is not None else self.sparse_weight
        )
        use_min_score = (
            min_hybrid_score if min_hybrid_score is not None else self.min_hybrid_score
        )

        # Adaptive k: good balance for small ↔ large collections
        n_docs = len(self.documents)
        adaptive_k = max(9.0, min(60.0, math.log2(n_docs + 1) * 5))
        actual_k = adaptive_k if use_k == 60.0 else use_k  # respect explicit override

        self.logger.debug(
            f"Using RRF k={actual_k:.1f} (adaptive), "
            f"dense_weight={use_dense_w}, sparse_weight={use_sparse_w}"
        )

        dense_k = dense_top_k if dense_top_k is not None else top_k * 3
        sparse_k = sparse_top_k if sparse_top_k is not None else top_k * 3

        # ────── Retrieve ──────
        dense_results = self.vector_search.search(
            query=query,
            documents=self.documents,
            ids=self.ids,
            top_k=dense_k,
            **search_kwargs,
        )

        sparse_results = self._sparse_search(query, top_k=sparse_k)

        # ────── Optional debug output ──────
        should_debug = debug or (self.logger and self.logger.level <= 10)
        if should_debug:
            self.logger.debug("Top Dense (pre-fusion):")
            for r in dense_results[:5]:
                self.logger.debug(f"  {r.get('score', 0.0):.3f}  {r['text'][:68]}...")

            self.logger.debug("Top Sparse (pre-fusion):")
            for r in sparse_results[:5]:
                self.logger.debug(f"  {r.get('score', 0.0):.3f}  {r['text'][:68]}...")

        # Fusion
        fused = reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            k=actual_k,
            limit=top_k,
            dense_weight=use_dense_w,
            sparse_weight=use_sparse_w,
        )

        if not fused:
            self.logger.warning(f"No results after fusion for query: {query!r}")
            return []

        # Filter weak blended results
        fused = [r for r in fused if r["hybrid_score"] >= use_min_score]

        # Optional: log distribution of categories for debugging
        if should_debug:
            from collections import Counter

            cats = Counter(r["category"] for r in fused)
            self.logger.debug(f"Result categories: {dict(cats)}")

        return fused


# ────────────────────────────────────────────────
#                   Example Usage
# ────────────────────────────────────────────────

if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table

    model: LLAMACPP_EMBED_KEYS = "nomic-embed-text"

    docs_data = [
        {
            "id": "d1",
            "content": "Hybrid vector search best practices 2025. Use RRF for combining BM25 and dense embeddings. Run both retrievers in parallel and fuse with reciprocal rank fusion...",
        },
        {
            "id": "d2",
            "content": "nomic-embed-text-v1.5 performance. Very fast on llama.cpp especially with Q5_K_M quantization. Low memory usage and excellent latency for local inference...",
        },
        {
            "id": "d3",
            "content": "Reciprocal Rank Fusion. Simple yet powerful fusion method used in Elastic, Weaviate, Azure Search, and many production RAG systems...",
        },
        {
            "id": "d4",
            "content": "Local embedding servers. llama.cpp provides OpenAI compatible API for embedding models like nomic-embed-text-v1.5. Easy to run on CPU/GPU...",
        },
        {
            "id": "d5",
            "content": "BM25 is still very strong. Especially good at rare terms, IDs, exact matches, product codes, and keyword precision in hybrid search...",
        },
    ]

    ids = [doc["id"] for doc in docs_data]
    documents = [doc["content"] for doc in docs_data]

    hybrid = HybridSearch.from_documents(
        documents=documents,
        ids=ids,
        model=model,
    )

    query = "fast local embeddings with llama.cpp"
    results = hybrid.search(query, top_k=5)

    console = Console()
    table = Table(title=f"Hybrid Results for: {query!r}")
    table.add_column("Rank", justify="right", style="cyan")
    table.add_column("Hybrid", justify="right")
    table.add_column("Dense", justify="right")
    table.add_column("Sparse", justify="right")
    table.add_column("Category", style="bold")
    table.add_column("Level", justify="right", style="dim cyan")
    table.add_column("ID")
    table.add_column("Preview", style="dim")

    for res in results:
        preview = res["text"][:80] + "..." if len(res["text"]) > 80 else res["text"]
        table.add_row(
            str(res["rank"]),
            f"{res['hybrid_score']:.4f}",
            f"{res['dense_score']:.3f}",
            f"{res['sparse_score']:.3f}",
            res["category"],
            str(res["category_level"]),
            res["id"] or "-",
            preview,
        )

    console.print(table)
