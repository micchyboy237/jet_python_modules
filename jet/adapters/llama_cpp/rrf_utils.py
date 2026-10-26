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

from rich.console import Console
from rich.table import Table

console = Console()


class RRFResult(TypedDict):
    """Standard fused result – matches vector/rerank shape + extra fields."""

    rank: int
    index: int
    score: float  # RRF score
    text: str
    sources: list[str]  # which lists contributed (e.g. ["dense", "bm25"])
    original_ranks: dict[str, int]  # list_name → rank


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    *,
    k: int = 60,
    weights: list[float] | None = None,
    top_n: int | None = None,
    list_names: list[str] | None = None,
    id_key: Literal["index", "text", "id"] = "index",
) -> list[RRFResult]:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        ranked_lists: List of ranked result lists.
                      Each inner list must contain dicts with at least
                      ``text`` and preferably ``index``.
        k: Smoothing constant (industry default = 60).
        weights: Optional per-list weights (same length as ranked_lists).
                 Defaults to 1.0 for every list.
        top_n: Keep only the top-N after fusion.
        list_names: Human-readable names for the lists (used in ``sources``).
                    Defaults to ["list_0", "list_1", ...].
        id_key: Key used to identify the same document across lists.
                - "index" → use original document index (preferred)
                - "text"  → use exact text match
                - "id"    → use an "id" field if present

    Returns:
        List of RRFResult sorted by descending RRF score.
    """
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
            else:  # "text"
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
    dense_top_n: int = 50,
    sparse_top_n: int = 50,
    final_top_n: int = 10,
    k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    use_rerank: bool = False,
    rerank_top_n: int | None = None,
) -> list[RRFResult]:
    """
    One-shot hybrid retrieval using dense + BM25 → RRF → optional cross-encoder.

    This is the recommended high-level entry point for most RAG pipelines.
    """
    from jet.adapters.llama_cpp.rerank_utils import rerank
    from jet.adapters.llama_cpp.vector_utils import vector_search
    from jet.vectors.reranker.bm25 import rerank_bm25

    if not documents:
        return []

    # 1. Dense retrieval
    dense_results = vector_search(
        query=query,
        documents=documents,
        top_n=dense_top_n,
    )

    # 2. Sparse (BM25) retrieval
    _, bm25_raw = rerank_bm25(query=query, documents=documents)
    sparse_results = [
        {
            "rank": r["rank"],
            "index": documents.index(r["text"]) if r["text"] in documents else -1,
            "score": r["score"],
            "text": r["text"],
        }
        for r in bm25_raw[:sparse_top_n]
    ]

    # ------------------------------------------------------------------
    # Improvement A: Debug logging of individual ranked lists (Rich)
    # ------------------------------------------------------------------
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

    # 3. RRF fusion
    fused = reciprocal_rank_fusion(
        ranked_lists=[dense_results, sparse_results],
        k=k,
        weights=[dense_weight, sparse_weight],
        list_names=["dense", "bm25"],
        top_n=rerank_top_n or final_top_n * 3,
        id_key="index",
    )

    if not use_rerank:
        return fused[:final_top_n]

    # 4. Optional cross-encoder re-ranking
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


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # 1. Low-level demo (manual ranked lists)
    # ------------------------------------------------------------------
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
        {"rank": 5, "index": 1, "score": 2.1, "text": docs[1]},
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

    # ------------------------------------------------------------------
    # 2. High-level hybrid demo
    # ------------------------------------------------------------------
    console.print("\n[bold cyan]2. High-level hybrid_rrf_search()[/bold cyan]")
    console.print("-" * 50)
    console.print(
        "[dim]Running dense + BM25 → RRF (no cross-encoder for speed)...[/dim]\n"
    )

    try:
        hybrid_results = hybrid_rrf_search(
            query=query,
            documents=docs,
            dense_top_n=5,
            sparse_top_n=5,
            final_top_n=5,
            use_rerank=False,
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
