# hybrid_search.py
import math
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

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
# Category configuration (flexible & reusable)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RelevanceCategory:
    label: str
    min_score: float  # absolute threshold (used in absolute mode)
    level: int  # 0 = worst, higher = better


@dataclass
class CategoryConfig:
    """
    Configurable mapping from hybrid score → (label, level).
    Supports absolute thresholds or relative percentiles.
    """

    categories: list[RelevanceCategory] = field(
        default_factory=lambda: [
            RelevanceCategory("Very Low", 0.000, 0),
            RelevanceCategory("Low", 0.025, 1),
            RelevanceCategory("Medium", 0.060, 2),
            RelevanceCategory("High", 0.085, 3),
            RelevanceCategory("Very High", 0.110, 4),
        ]
    )
    mode: Literal["absolute", "relative"] = "absolute"  # ← most important change
    relative_fractions: list[float] | None = None  # e.g. [0.0, 0.3, 0.5, 0.7, 0.9]

    def __post_init__(self):
        self.categories = sorted(self.categories, key=lambda c: c.min_score)

    def get_category(
        self, score: float, max_score_in_results: float = 1.0
    ) -> tuple[str, int]:
        """
        Returns (label, level) for the given score.

        In relative mode, score is normalized against max_score_in_results.
        """
        if self.mode == "relative" and self.relative_fractions:
            norm = score / max(1e-9, max_score_in_results)
            for i, frac in enumerate(reversed(self.relative_fractions)):
                if norm >= frac:
                    cat = self.categories[-1 - i]  # highest first
                    return cat.label, cat.level
            return self.categories[0].label, self.categories[0].level

        # Absolute mode
        for cat in reversed(self.categories):
            if score >= cat.min_score:
                return cat.label, cat.level
        return self.categories[0].label, self.categories[0].level


# Predefined convenient configs
ABSOLUTE_CATEGORY_CONFIG = CategoryConfig(mode="absolute")

RELATIVE_CATEGORY_CONFIG = CategoryConfig(
    mode="relative",
    relative_fractions=[0.00, 0.40, 0.68, 0.85, 0.96],
    categories=[
        RelevanceCategory("Very Low", 0.000, 0),
        RelevanceCategory("Low", 0.025, 1),  # ← added floor
        RelevanceCategory("Medium", 0.065, 2),
        RelevanceCategory("High", 0.105, 3),
        RelevanceCategory("Very High", 0.145, 4),
    ],
)


# ──────────────────────────────────────────────────────────────────────────────
# Globals / Helpers
# ──────────────────────────────────────────────────────────────────────────────

stop_words = set(stopwords.words("english"))


def better_tokenize(text: str) -> list[str]:
    tokens = word_tokenize(text.lower())
    return [t for t in tokens if t.isalnum() and t not in stop_words]


# ──────────────────────────────────────────────────────────────────────────────
# Result type
# ──────────────────────────────────────────────────────────────────────────────


class HybridSearchResult(TypedDict):
    rank: int
    index: int
    id: str | None
    text: str
    dense_score: float
    sparse_score: float
    hybrid_score: float
    category: str
    category_level: int
    normalized_hybrid: float | None  # only present if normalization enabled


# ──────────────────────────────────────────────────────────────────────────────
# Fusion function
# ──────────────────────────────────────────────────────────────────────────────


def reciprocal_rank_fusion(
    dense_results: list[SearchResultType],
    sparse_results: list[SearchResultType],
    k: float = 60.0,
    limit: int | None = None,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> list[dict]:
    """
    Weighted Reciprocal Rank Fusion.
    Returns list of dicts with raw scores (no category yet).
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
            }
        )

    return final


# ──────────────────────────────────────────────────────────────────────────────
# Main HybridSearch class
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class HybridSearch:
    """
    Hybrid (dense + sparse) retriever using Weighted Reciprocal Rank Fusion.
    """

    documents: list[str]
    ids: list[str] | None = None
    embedding_model: LlamacppEmbedding | str = "nomic-embed-text"
    bm25: BM25Okapi | None = None
    vector_search: VectorSearch | None = None

    fusion_k: float = 12.0  # base value — adapted in search()
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    min_hybrid_score: float = 0.015
    category_config: CategoryConfig = field(
        default_factory=lambda: ABSOLUTE_CATEGORY_CONFIG
    )

    logger: CustomLogger | None = None

    def __post_init__(self):
        if self.logger is None:
            self.logger = CustomLogger()

        tokenized_docs = [better_tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)

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
        top_k: int | None = None,  # ← changed: now optional + None
        dense_top_k: int | None = None,
        sparse_top_k: int | None = None,
        fusion_k: float | None = None,
        dense_weight: float | None = None,
        sparse_weight: float | None = None,
        min_hybrid_score: float | None = None,
        category_config: CategoryConfig | None = None,
        normalize_scores: bool = False,
        debug: bool = False,
        **search_kwargs: Any,
    ) -> list[HybridSearchResult]:
        """
        Perform hybrid search.

        Args:
            top_k: Maximum number of results to return.
                   If None, returns all fused & ranked results (after min_hybrid_score filter).
            ...
        """
        use_k = fusion_k if fusion_k is not None else self.fusion_k
        use_dense_w = dense_weight if dense_weight is not None else self.dense_weight
        use_sparse_w = (
            sparse_weight if sparse_weight is not None else self.sparse_weight
        )
        use_min_score = (
            min_hybrid_score if min_hybrid_score is not None else self.min_hybrid_score
        )
        use_category_config = category_config or self.category_config

        n_docs = len(self.documents)
        adaptive_k = max(8.0, min(40.0, math.sqrt(n_docs) * 3.5))
        actual_k = adaptive_k if use_k == 60.0 else use_k

        self.logger.debug(
            f"Using RRF k={actual_k:.1f} (adaptive), "
            f"dense_weight={use_dense_w:.2f}, sparse_weight={use_sparse_w:.2f}"
        )

        # Decide how many candidates to retrieve before fusion
        # When top_k is None → we want to be more inclusive
        if top_k is None:
            # Get more candidates when returning all results
            dense_k = dense_top_k if dense_top_k is not None else max(100, n_docs // 2)
            sparse_k = (
                sparse_top_k if sparse_top_k is not None else max(100, n_docs // 2)
            )
        else:
            # Original behavior — fetch more candidates than final top_k
            dense_k = dense_top_k if dense_top_k is not None else top_k * 3
            sparse_k = sparse_top_k if sparse_top_k is not None else top_k * 3

        dense_results = self.vector_search.search(
            query=query,
            documents=self.documents,
            ids=self.ids,
            top_k=dense_k,
            **search_kwargs,
        )

        sparse_results = self._sparse_search(query, top_k=sparse_k)

        should_debug = debug or (self.logger and self.logger.level <= 10)
        if should_debug:
            self.logger.debug("Top Dense (pre-fusion):")
            for r in dense_results[:5]:
                self.logger.debug(f" {r.get('score', 0.0):.3f} {r['text'][:68]}...")
            self.logger.debug("Top Sparse (pre-fusion):")
            for r in sparse_results[:5]:
                self.logger.debug(f" {r.get('score', 0.0):.3f} {r['text'][:68]}...")

        # ────────────────────────────────────────────────
        # Main change here
        # ────────────────────────────────────────────────
        fused_raw = reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            k=actual_k,
            limit=top_k,  # ← pass top_k directly (can be None)
            dense_weight=use_dense_w,
            sparse_weight=use_sparse_w,
        )

        if not fused_raw:
            self.logger.warning(f"No results after fusion for query: {query!r}")
            return []

        max_hybrid = max((r["hybrid_score"] for r in fused_raw), default=0.0)

        fused: list[HybridSearchResult] = []
        for raw in fused_raw:
            label, level = use_category_config.get_category(
                raw["hybrid_score"],
                max_score_in_results=max_hybrid
                if use_category_config.mode == "relative"
                else 1.0,
            )
            result: HybridSearchResult = {
                **raw,
                "category": label,
                "category_level": level,
            }
            fused.append(result)

        if normalize_scores:
            max_h = max((r["hybrid_score"] for r in fused), default=1.0)
            for r in fused:
                r["normalized_hybrid"] = round(r["hybrid_score"] / max_h, 4)

        # Apply minimum score filter
        fused = [r for r in fused if r["hybrid_score"] >= use_min_score]

        if should_debug:
            from collections import Counter

            cats = Counter(r["category"] for r in fused)
            self.logger.debug(f"Result categories: {dict(cats)}")
            self.logger.debug(f"Final result count: {len(fused)}")

        return fused


# ────────────────────────────────────────────────
#                   Example Usage
# ────────────────────────────────────────────────

if __name__ == "__main__":
    from pathlib import Path

    from jet.code.extraction.html_sentence_extractor import html_to_sentences
    from jet.file.utils import save_file
    from rich.console import Console
    from rich.table import Table

    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem

    model: LLAMACPP_EMBED_KEYS = "nomic-embed-text"

    # docs_data = [
    #     {
    #         "id": "d1",
    #         "content": "Hybrid vector search best practices 2025. Use RRF for combining BM25 and dense embeddings. Run both retrievers in parallel and fuse with reciprocal rank fusion...",
    #     },
    #     {
    #         "id": "d2",
    #         "content": "nomic-embed-text-v1.5 performance. Very fast on llama.cpp especially with Q5_K_M quantization. Low memory usage and excellent latency for local inference...",
    #     },
    #     {
    #         "id": "d3",
    #         "content": "Reciprocal Rank Fusion. Simple yet powerful fusion method used in Elastic, Weaviate, Azure Search, and many production RAG systems...",
    #     },
    #     {
    #         "id": "d4",
    #         "content": "Local embedding servers. llama.cpp provides OpenAI compatible API for embedding models like nomic-embed-text-v1.5. Easy to run on CPU/GPU...",
    #     },
    #     {
    #         "id": "d5",
    #         "content": "BM25 is still very strong. Especially good at rare terms, IDs, exact matches, product codes, and keyword precision in hybrid search...",
    #     },
    # ]

    # ids = [doc["id"] for doc in docs_data]
    # documents = [doc["content"] for doc in docs_data]

    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem

    # Load HTML content from the specified file
    html_file_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/vectors/semantic_search/generated/run_web_search/top_isekai_anime_2026/pages/gamerant_com_new_isekai_anime_2026/page.html"
    with open(html_file_path, encoding="utf-8") as file:
        html = file.read()

    sentences = html_to_sentences(html)
    documents = sentences

    hybrid = HybridSearch.from_documents(
        documents=documents,
        # ids=ids,
        model=model,
        # Example: use relative mode instead
        # category_config=RELATIVE_CATEGORY_CONFIG
    )

    query = "Top isekai anime 2026"
    results = hybrid.search(
        query,
        top_k=None,
        normalize_scores=True,  # adds normalized_hybrid field
        debug=True,
    )

    console = Console()
    table = Table(title=f"Hybrid Results for: {query!r}")
    table.add_column("Rank", justify="right", style="cyan")
    table.add_column("Hybrid", justify="right")
    table.add_column("Dense", justify="right")
    table.add_column("Sparse", justify="right")
    table.add_column("Norm", justify="right", style="green")
    table.add_column("Category", style="bold")
    # table.add_column("Level", justify="right", style="dim cyan")
    # table.add_column("ID")
    table.add_column("Preview", style="dim")

    for res in results[:10]:
        preview = res["text"][:80] + "..." if len(res["text"]) > 80 else res["text"]
        norm_str = (
            f"{res.get('normalized_hybrid', '-'):.3f}"
            if "normalized_hybrid" in res
            else "-"
        )
        table.add_row(
            str(res["rank"]),
            f"{res['hybrid_score']:.4f}",
            f"{res['dense_score']:.3f}",
            f"{res['sparse_score']:.3f}",
            norm_str,
            res["category"],
            # str(res["category_level"]),
            # res["id"] or "-",
            preview,
        )

    console.print(table)

    save_file(results, OUTPUT_DIR / "results.json")
