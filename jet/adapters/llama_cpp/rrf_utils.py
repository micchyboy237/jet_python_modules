"""Reciprocal Rank Fusion (RRF) utilities.
Compatible with the result shapes produced by:
- jet.adapters.llama_cpp.vector_utils.vector_search
- jet.adapters.llama_cpp.rerank_utils.rerank
- jet.vectors.reranker.bm25.rerank_bm25
Formula:
    RRF_score(d) = Σ weight_i / (k + rank_i(d))
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal, TypedDict

import numpy as np
from jet.adapters.llama_cpp.config import EMBED_MODEL
from rich.console import Console
from rich.table import Table

console = Console()


class RRFResult(TypedDict):
    """Standard fused result – matches vector/rerank shape + extra fields."""

    rank: int
    index: int
    score: float
    text: str
    sources: list[str]
    original_ranks: dict[str, int]


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    *,
    k: int = 60,
    weights: list[float] | None = None,
    top_n: int | None = None,
    list_names: list[str] | None = None,
    id_key: Literal["index", "text", "id"] = "index",
) -> list[RRFResult]:
    # ... [Existing reciprocal_rank_fusion code remains unchanged] ...
    if not ranked_lists:
        console.print("[yellow]Warning: No ranked lists provided to RRF[/yellow]")
        return []

    n_lists = len(ranked_lists)
    weights = weights or [1.0] * n_lists
    if len(weights) != n_lists:
        raise ValueError(
            f"weights length ({len(weights)}) must match number of lists ({n_lists})"
        )

    list_names = list_names or [f"list_{i}" for i in range(n_lists)]
    if len(list_names) != n_lists:
        raise ValueError("list_names length must match number of lists")

    score_map: dict[Any, float] = defaultdict(float)
    doc_map: dict[Any, dict[str, Any]] = {}
    sources_map: dict[Any, list[str]] = defaultdict(list)
    ranks_map: dict[Any, dict[str, int]] = defaultdict(dict)

    for list_idx, ranked in enumerate(ranked_lists):
        weight = weights[list_idx]
        name = list_names[list_idx]
        for rank, item in enumerate(ranked, start=1):
            if id_key == "index":
                doc_id = item.get("index")
                if doc_id is None:
                    doc_id = item.get("text")
            elif id_key == "id":
                doc_id = item.get("id") or item.get("index") or item.get("text")
            else:
                doc_id = item.get("text")

            if doc_id is None:
                console.print(
                    f"[yellow]Skipping item without identifiable key: {item}[/yellow]"
                )
                continue

            contribution = weight * (1.0 / (k + rank))
            score_map[doc_id] += contribution
            if doc_id not in doc_map:
                doc_map[doc_id] = item
            sources_map[doc_id].append(name)
            ranks_map[doc_id][name] = rank

    fused: list[RRFResult] = []
    for doc_id, rrf_score in score_map.items():
        original = doc_map[doc_id]
        fused.append(
            {
                "rank": 0,
                "index": original.get("index", -1),
                "score": rrf_score,
                "text": original.get("text", ""),
                "sources": sources_map[doc_id],
                "original_ranks": ranks_map[doc_id],
            }
        )

    fused.sort(key=lambda x: x["score"], reverse=True)
    for rank, item in enumerate(fused, start=1):
        item["rank"] = rank

    if top_n is not None:
        fused = fused[:top_n]

    console.print(
        f"[green]RRF fused {len(score_map)} unique documents from {n_lists} lists "
        f"→ returning {len(fused)}[/green]"
    )
    return fused


def hybrid_rrf_search(
    query: str,
    documents: list[str],
    *,
    dense_top_n: int = 20,
    sparse_top_n: int = 20,
    final_top_n: int = 10,
    k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    use_rerank: bool = False,
    rerank_top_n: int | None = None,
    sparse_min_score: float = 0.01,
    embed_model: str = EMBED_MODEL,
    # NEW ARGS FOR PRE-COMPUTED EMBEDDINGS
    dense_embeddings: np.ndarray | None = None,
    dense_indices: list[int] | None = None,
) -> list[RRFResult]:
    """
    One-shot hybrid retrieval using dense + BM25 → RRF → optional cross-encoder.

    Args:
        query: Search query.
        documents: List of document strings.
        dense_top_n: Number of candidates from dense search.
        sparse_top_n: Number of candidates from BM25 search.
        final_top_n: Final number of results to return.
        k: RRF smoothing constant.
        dense_weight: Weight for dense results in RRF.
        sparse_weight: Weight for sparse results in RRF.
        use_rerank: Whether to apply cross-encoder reranking after fusion.
        rerank_top_n: Number of candidates to pass to reranker.
        sparse_min_score: Minimum BM25 score to include in fusion.
        embed_model: The embedding model identifier.
        dense_embeddings: Optional pre-computed embeddings for 'documents'.
        dense_indices: Optional list of original indices corresponding to 'dense_embeddings'.
                       If provided, 'documents' should still contain all texts for BM25,
                       but dense search will only use the subset defined by these indices.
    """
    from jet.adapters.llama_cpp.embed_utils import embed
    from jet.adapters.llama_cpp.rerank_utils import rerank
    from jet.adapters.llama_cpp.vector_utils import cosine_similarity, vector_search
    from jet.vectors.reranker.bm25 import rerank_bm25

    if not documents:
        return []

    # 1. Dense Search Path
    if dense_embeddings is not None and dense_indices is not None:
        console.print("[dim]Using pre-computed embeddings for dense search...[/dim]")
        # Embed query
        query_emb = embed(query, model=embed_model)
        if isinstance(query_emb, list):
            query_emb = np.array(query_emb, dtype=np.float32)

        # Calculate similarity only for the pre-selected indices
        selected_embs = dense_embeddings
        similarities = cosine_similarity(query_emb, selected_embs)

        # Create dense results list
        dense_results_raw = []
        for local_idx, score in enumerate(similarities):
            original_idx = dense_indices[local_idx]
            dense_results_raw.append(
                {
                    "index": original_idx,
                    "score": float(score),
                    "text": documents[original_idx],
                }
            )

        # Sort by score descending
        dense_results_raw.sort(key=lambda x: x["score"], reverse=True)

        # Assign ranks and take top_n
        dense_results = []
        for rank, item in enumerate(dense_results_raw[:dense_top_n], start=1):
            item["rank"] = rank
            dense_results.append(item)

    else:
        # Fallback to standard vector_search if no pre-computed embeddings
        dense_results = vector_search(
            query=query,
            documents=documents,
            top_n=dense_top_n,
            embed_model=embed_model,
        )

    # 2. Sparse Search Path (BM25)
    _, bm25_raw = rerank_bm25(query=query, documents=documents)

    sparse_results = []
    for r in bm25_raw[:sparse_top_n]:
        if r["score"] < sparse_min_score:
            continue
        try:
            idx = documents.index(r["text"])
        except ValueError:
            idx = -1
        sparse_results.append(
            {
                "rank": r["rank"],
                "index": idx,
                "score": r["score"],
                "text": r["text"],
            }
        )

    console.print("\n[bold cyan]Dense top results (before RRF):[/bold cyan]")
    for r in dense_results[: min(8, len(dense_results))]:
        console.print(
            f"  [cyan]#{r['rank']:<2}[/cyan]  "
            f"idx=[green]{r['index']:<3}[/green]  "
            f"score=[yellow]{r['score']:.4f}[/yellow]  "
            f"{r['text'][:70]}"
        )

    console.print("\n[bold cyan]BM25 top results (before RRF):[/bold cyan]")
    for r in sparse_results[: min(8, len(sparse_results))]:
        console.print(
            f"  [cyan]#{r['rank']:<2}[/cyan]  "
            f"idx=[green]{r['index']:<3}[/green]  "
            f"score=[yellow]{r['score']:.4f}[/yellow]  "
            f"{r['text'][:70]}"
        )
    console.print()

    fused = reciprocal_rank_fusion(
        ranked_lists=[dense_results, sparse_results],
        k=k,
        weights=[dense_weight, sparse_weight],
        list_names=["dense", "bm25"],
        top_n=rerank_top_n or final_top_n * 3,
        id_key="text",
    )

    if not use_rerank:
        return fused[:final_top_n]

    candidate_texts = [r["text"] for r in fused]
    reranked = rerank(
        query=query,
        documents=candidate_texts,
        top_n=final_top_n,
        method="auto",
    )

    final: list[RRFResult] = []
    for r in reranked:
        matched = next((f for f in fused if f["text"] == r["text"]), None)
        final.append(
            {
                "rank": r["rank"],
                "index": matched["index"] if matched else r["index"],
                "score": r["score"],
                "text": r["text"],
                "sources": matched["sources"] if matched else ["rerank"],
                "original_ranks": matched["original_ranks"] if matched else {},
            }
        )
    return final


if __name__ == "__main__":
    query = "What is a giant panda?"
    docs = [
        "The giant panda is a bear species endemic to China.",
        "Python is a high-level programming language.",
        "Bears are carnivoran mammals of the family Ursidae.",
        "Machine learning is a subset of artificial intelligence.",
        "Pandas eat bamboo and live in mountainous regions.",
        "The red panda is a small mammal native to the eastern Himalayas.",
        "Bamboo is the primary food source for giant pandas.",
    ]

    console.print("\n[bold blue]" + "=" * 70)
    console.print("[bold blue]RRF Utils Demo[/bold blue]")
    console.print("[bold blue]" + "=" * 70)
    console.print(f"\n[bold]Query:[/bold] {query}\n")

    console.print("[bold cyan]1. Low-level reciprocal_rank_fusion()[/bold cyan]")
    console.print("-" * 50)
    dense_list = [
        {"rank": 1, "index": 0, "score": 0.92, "text": docs[0]},
        {"rank": 2, "index": 4, "score": 0.85, "text": docs[4]},
        {"rank": 3, "index": 6, "score": 0.78, "text": docs[6]},
        {"rank": 4, "index": 2, "score": 0.71, "text": docs[2]},
        {"rank": 5, "index": 5, "score": 0.65, "text": docs[5]},
    ]
    sparse_list = [
        {"rank": 1, "index": 0, "score": 12.4, "text": docs[0]},
        {"rank": 2, "index": 6, "score": 9.8, "text": docs[6]},
        {"rank": 3, "index": 4, "score": 8.1, "text": docs[4]},
        {"rank": 4, "index": 2, "score": 5.3, "text": docs[2]},
        {"rank": 5, "index": 1, "score": 0.0, "text": docs[1]},
    ]
    fused = reciprocal_rank_fusion(
        ranked_lists=[dense_list, sparse_list],
        k=60,
        weights=[1.0, 1.0],
        list_names=["dense", "bm25"],
        top_n=5,
        id_key="index",
    )

    table = Table(
        title="RRF Fused Results", show_header=True, header_style="bold magenta"
    )
    table.add_column("Rank", style="cyan", width=6)
    table.add_column("Index", style="green", width=6)
    table.add_column("RRF Score", style="yellow", width=12)
    table.add_column("Sources", style="blue", width=18)
    table.add_column("Original Ranks", style="dim", width=22)
    table.add_column("Text", style="white")
    for r in fused:
        table.add_row(
            str(r["rank"]),
            str(r["index"]),
            f"{r['score']:.5f}",
            ", ".join(r["sources"]),
            str(r["original_ranks"]),
            r["text"][:55] + ("..." if len(r["text"]) > 55 else ""),
        )
    console.print(table)

    console.print("\n[bold cyan]2. High-level hybrid_rrf_search()[/bold cyan]")
    console.print("-" * 50)
    console.print(
        "[dim]Running dense + BM25 → RRF (no cross-encoder for speed)...[/dim]\n"
    )
    try:
        hybrid_results = hybrid_rrf_search(
            query=query,
            documents=docs,
            dense_top_n=20,
            sparse_top_n=20,
            final_top_n=5,
            use_rerank=False,
            sparse_min_score=0.01,
        )
        table2 = Table(
            title="Hybrid RRF Search Results",
            show_header=True,
            header_style="bold magenta",
        )
        table2.add_column("Rank", style="cyan", width=6)
        table2.add_column("Index", style="green", width=6)
        table2.add_column("Score", style="yellow", width=10)
        table2.add_column("Sources", style="blue", width=18)
        table2.add_column("Text", style="white")
        for r in hybrid_results:
            table2.add_row(
                str(r["rank"]),
                str(r["index"]),
                f"{r['score']:.5f}",
                ", ".join(r["sources"]),
                r["text"][:60] + ("..." if len(r["text"]) > 60 else ""),
            )
        console.print(table2)
    except Exception as e:
        console.print(f"[red]Hybrid search failed (servers may be offline): {e}[/red]")
        console.print(
            "[yellow]The low-level RRF demo above still works without servers.[/yellow]"
        )

    console.print("\n[bold green]Demo finished.[/bold green]")
