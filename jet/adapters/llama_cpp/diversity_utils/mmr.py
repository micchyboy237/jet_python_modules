"""
MMR (Maximal Marginal Relevance) – Query-Focused Diverse Retrieval
==================================================================

Description:
    This module implements Maximal Marginal Relevance (MMR) to rerank
    retrieval results, balancing relevance to the query with diversity
    among selected items. It prevents redundant near-duplicates from
    dominating the top-K results.

    MMR Formula:
        MMR = argmax [ λ * sim(doc, query) - (1 - λ) * max(sim(doc, selected)) ]

    Where:
        - λ (lambda): Trade-off parameter (1.0 = pure relevance, 0.0 = pure diversity)
        - sim(doc, query): Cosine similarity between document and query
        - sim(doc, selected): Maximum cosine similarity between document and any already selected document

Usage Examples:
    1. Basic usage with auto-query (corpus centroid):
       python mmr.py

    2. Specific query with custom lambda:
       python mmr.py -q "machine learning algorithms" -l 0.7 -k 10

    3. Using a custom samples file:
       python mmr.py /path/to/samples.json -q "python programming" -k 5

Resources:
    - MMR Original Paper: Carbonell & Goldstein, SIGIR 1998
    - Elastic Search Labs: https://www.elastic.co/search-labs/blog/maximum-marginal-relevance-diversify-results
    - VizLearn MMR Guide: https://vizlearn.in/gen_ai/maximal_marginal_relevance.html
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import TypedDict

import numpy as np
from jet.adapters.llama_cpp.config import (
    EMBED_DOC_PREFIX,
    EMBED_MODEL,
    EMBED_QUERY_PREFIX,
)
from jet.adapters.llama_cpp.embed_utils import embed
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.feature_extraction.text import TfidfVectorizer

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem


class SampleItem(TypedDict):
    """Represents a single sample item from the JSON data."""

    code: str
    text: str
    videoId: str


class MMRResult(TypedDict):
    """Represents a single MMR selection result."""

    index: int
    relevance: float
    redundancy: float
    mmr_score: float


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def extract_top_keywords(texts: list[str], n: int = 6) -> str:
    """
    Return the top-n TF-IDF keywords from a list of texts as a
    comma-separated string. Used only for the auto-query display label.
    """
    if not texts:
        return ""
    vectorizer = TfidfVectorizer(
        max_features=n * 4,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
        top_indices = scores.argsort()[-n:][::-1]
        terms = vectorizer.get_feature_names_out()
        return ", ".join(terms[i] for i in top_indices)
    except Exception:
        return "N/A"


def mmr_rerank(
    query_vec: np.ndarray,
    doc_vecs: np.ndarray,
    top_k: int,
    lambda_param: float,
) -> list[MMRResult]:
    """
    Maximal Marginal Relevance selection.

    Parameters
    ----------
    query_vec     : shape (dim,)   – embedded query (or corpus centroid)
    doc_vecs      : shape (n, dim) – embedded documents
    top_k         : how many items to pick
    lambda_param  : λ — relevance vs. diversity weight

    Returns
    -------
    List of dicts ordered by selection, each containing:
        index       – original index into doc_vecs / data
        relevance   – cosine similarity to query vector
        redundancy  – max cosine similarity to any already-selected doc
        mmr_score   – final MMR score used to pick this item
    """
    n = len(doc_vecs)
    if n == 0:
        return []

    top_k = min(top_k, n)

    # Precompute relevance scores for all documents
    relevance_scores = np.array(
        [cosine_similarity(query_vec, doc_vecs[i]) for i in range(n)]
    )

    selected: list[int] = []
    results: list[MMRResult] = []
    remaining = list(range(n))

    for _ in range(top_k):
        best_idx = None
        best_score = -np.inf
        best_rel = 0.0
        best_red = 0.0

        for idx in remaining:
            rel = relevance_scores[idx]

            if selected:
                # Calculate max similarity to any already selected document
                redundancy = max(
                    cosine_similarity(doc_vecs[idx], doc_vecs[s]) for s in selected
                )
            else:
                redundancy = 0.0

            # MMR formula
            score = lambda_param * rel - (1 - lambda_param) * redundancy

            if score > best_score:
                best_score = score
                best_idx = idx
                best_rel = rel
                best_red = redundancy

        if best_idx is None:
            break

        selected.append(best_idx)
        remaining.remove(best_idx)

        results.append(
            {
                "index": best_idx,
                "relevance": round(best_rel, 4),
                "redundancy": round(best_red, 4),
                "mmr_score": round(best_score, 4),
            }
        )

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "MMR-based diverse retrieval: re-ranks similarity results so the "
            "final selection is both relevant AND non-redundant.\n\n"
            "When --query is omitted the corpus embedding centroid is used "
            "automatically, and top TF-IDF keywords are printed as a label."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "samples_path",
        nargs="?",
        default=str(Path(__file__).parent / "mocks" / "mmr_samples.json"),
        help="Path to the samples JSON file.",
    )
    parser.add_argument(
        "-q",
        "--query",
        default=None,
        help=(
            "Natural-language query to retrieve against. "
            "Omit to auto-derive from the corpus centroid."
        ),
    )
    parser.add_argument(
        "-k",
        "--top_k",
        type=int,
        default=5,
        help="Number of results to return.",
    )
    parser.add_argument(
        "-l",
        "--lambda_param",
        type=float,
        default=0.5,
        help=(
            "MMR trade-off weight λ ∈ [0, 1]. 1 = pure relevance, 0 = pure diversity."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory to save results",
    )
    parser.add_argument(
        "--auto_keywords",
        type=int,
        default=6,
        metavar="N",
        help="Number of TF-IDF keywords to show when query is auto-derived.",
    )
    return parser.parse_args()


def save_results(
    mmr_results: list[MMRResult], docs: list[SampleItem], args: argparse.Namespace
):
    """Save MMR results and metadata to output_dir."""
    output_data = {
        "query_mode": "explicit" if args.query else "auto (corpus centroid)",
        "query": args.query or "[auto-derived]",
        "lambda": args.lambda_param,
        "top_k": args.top_k,
        "total_documents": len(docs),
        "results": [
            {
                **result,
                "videoId": docs[result["index"]]["videoId"],
                "text": docs[result["index"]]["text"],
                "code": docs[result["index"]]["code"],
            }
            for result in mmr_results
        ],
    }

    results_file = args.output_dir / "mmr_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    console.print(f"[green]✓[/green] Results saved to: {results_file}")


def main():
    args = parse_args()

    shutil.rmtree(args.output_dir, ignore_errors=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples_path = Path(args.samples_path)

    if not samples_path.exists():
        console.print(f"[red]Error:[/red] Samples file not found: {samples_path}")
        return

    try:
        with open(samples_path, "r", encoding="utf-8") as f:
            docs = json.load(f)
    except json.JSONDecodeError as e:
        console.print(f"[red]Error:[/red] Invalid JSON in {samples_path}: {e}")
        return

    if not docs:
        console.print("[yellow]Warning:[/yellow] No documents loaded.")
        return

    console.print(
        Panel.fit(
            f"[bold]MMR Diverse Retrieval[/bold]\n"
            f"Loaded [cyan]{len(docs)}[/cyan] samples from [italic]{samples_path.name}[/italic]",
            border_style="blue",
        )
    )

    # Embed documents
    console.print("[dim]⏳ Embedding documents...[/dim]")

    # Extract text fields for embedding if docs are objects
    if isinstance(docs[0], dict):
        texts_to_embed = [doc.get("text", "") for doc in docs]
    else:
        texts_to_embed = docs

    try:
        doc_vecs = embed(texts_to_embed, model=EMBED_MODEL, prefix=EMBED_DOC_PREFIX)
    except Exception as e:
        console.print(f"[red]Error embedding documents:[/red] {e}")
        return

    if len(doc_vecs) != len(docs):
        console.print("[yellow]Warning:[/yellow] Embedding count mismatch.")
        return

    console.print("[green]✓[/green] Document embeddings ready.\n")

    # Handle query
    if args.query:
        query_label = f'"{args.query}"'
        query_mode = "explicit"
        console.print("[dim]⏳ Embedding query...[/dim]")
        try:
            query_vec = embed(args.query, model=EMBED_MODEL, prefix=EMBED_QUERY_PREFIX)
        except Exception as e:
            console.print(f"[red]Error embedding query:[/red] {e}")
            return
        console.print("[green]✓[/green] Query embedding ready.\n")
    else:
        query_vec = doc_vecs.mean(axis=0)
        query_mode = "auto (corpus centroid)"
        auto_kws = extract_top_keywords([d["text"] for d in docs], n=args.auto_keywords)
        query_label = f"[auto: {auto_kws}]"

    # Display configuration
    lambda_desc = (
        "pure relevance"
        if args.lambda_param == 1.0
        else "pure diversity"
        if args.lambda_param == 0.0
        else "balanced"
    )

    console.print(
        Panel.fit(
            f"[bold]Configuration[/bold]\n"
            f"Query Mode : {query_mode}\n"
            f"Query      : {query_label}\n"
            f"λ (lambda) : {args.lambda_param} ({lambda_desc})\n"
            f"Top-k      : {args.top_k}",
            border_style="green",
        )
    )

    # Perform MMR reranking
    console.print("[dim]⏳ Running MMR reranking...[/dim]")
    mmr_results = mmr_rerank(
        query_vec=query_vec,
        doc_vecs=doc_vecs,
        top_k=args.top_k,
        lambda_param=args.lambda_param,
    )

    if not mmr_results:
        console.print("[yellow]No results returned.[/yellow]")
        return

    # Display results in a rich table
    table = Table(title=f"MMR TOP-{args.top_k} Results (λ={args.lambda_param})")
    table.add_column("#", style="cyan", justify="right")
    table.add_column("MMR Score", justify="right")
    table.add_column("Relevance", justify="right")
    table.add_column("Redundancy", justify="right")
    table.add_column("Video ID")
    table.add_column("Text Preview", max_width=50)

    for rank, r in enumerate(mmr_results, 1):
        item = docs[r["index"]]
        table.add_row(
            str(rank),
            f"{r['mmr_score']:.4f}",
            f"{r['relevance']:.4f}",
            f"{r['redundancy']:.4f}",
            item["videoId"],
            item["text"][:50] + "..." if len(item["text"]) > 50 else item["text"],
        )

    console.print(table)

    # Save results
    save_results(mmr_results, docs, args)


if __name__ == "__main__":
    main()
