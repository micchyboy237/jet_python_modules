"""
Approach 5 – MMR (Maximal Marginal Relevance) Query-Focused Diverse Retrieval

Why this approach matters
────────────────────────────────────────────────────────────────────────
Plain similarity search returns the TOP-K most relevant items — but these
are often near-duplicates of each other, wasting result slots.

MMR fixes this by alternating between two forces on every pick:
  • Relevance  – how similar the candidate is to your query
  • Novelty    – how DIFFERENT the candidate is from items already selected

The trade-off is controlled by λ (lambda):
  λ = 1.0  →  pure relevance   (same as cosine top-k)
  λ = 0.0  →  pure diversity   (maximally spread-out picks)
  λ = 0.5  →  balanced default (recommended starting point)

This is the go-to technique in 2025/2026 for search result diversification,
RAG context selection, and recommendation deduplication.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from jet.adapters.llama_cpp.embed_utils import embed

# ── Prefixes used by the nomic / e5-family embedding models ──────────────────
QUERY_PREFIX = "search_query: "
DOC_PREFIX = "search_document: "


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "MMR-based diverse retrieval: re-ranks similarity results so the "
            "final selection is both relevant AND non-redundant."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "samples_path",
        nargs="?",
        default=str(Path(__file__).parent / "mocks" / "05_samples.json"),
        help="Path to the samples JSON file.",
    )
    parser.add_argument(
        "-q",
        "--query",
        default="cuckold husband watches wife with another man",
        help="Natural-language query to retrieve against.",
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
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def mmr_select(
    query_vec: np.ndarray,
    doc_vecs: np.ndarray,
    top_k: int,
    lambda_param: float,
) -> list[dict]:
    """
    Maximal Marginal Relevance selection.

    Parameters
    ----------
    query_vec     : shape (dim,)  – embedded query
    doc_vecs      : shape (n, dim) – embedded documents
    top_k         : how many items to pick
    lambda_param  : λ — relevance vs. diversity weight

    Returns
    -------
    List of dicts, each containing:
        index       – original index in doc_vecs / data
        relevance   – cosine similarity to query
        redundancy  – max cosine similarity to any already-selected doc
        mmr_score   – final MMR score used to pick this item
    """
    n = len(doc_vecs)
    top_k = min(top_k, n)

    # Pre-compute relevance for all docs (stays fixed throughout)
    relevance_scores = np.array(
        [cosine_similarity(query_vec, doc_vecs[i]) for i in range(n)]
    )

    selected: list[int] = []  # indices of already-chosen docs
    results: list[dict] = []

    remaining = list(range(n))  # indices still available to pick

    for _ in range(top_k):
        best_idx = None
        best_score = -np.inf

        for idx in remaining:
            rel = relevance_scores[idx]

            # Max similarity to any already-selected document
            if selected:
                redundancy = max(
                    cosine_similarity(doc_vecs[idx], doc_vecs[s]) for s in selected
                )
            else:
                redundancy = 0.0  # nothing selected yet → no penalty

            score = lambda_param * rel - (1 - lambda_param) * redundancy

            if score > best_score:
                best_score = score
                best_idx = idx
                best_rel = rel
                best_red = redundancy

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


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    # ── Load samples ──────────────────────────────────────────────────────────
    samples_path = Path(args.samples_path)
    with open(samples_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"📂 Loaded {len(data)} samples from: {samples_path.name}")
    print(f'🔍 Query      : "{args.query}"')
    print(
        f"🎛️  λ (lambda) : {args.lambda_param}  "
        f"({'pure relevance' if args.lambda_param == 1 else 'pure diversity' if args.lambda_param == 0 else 'balanced'})"
    )
    print(f"🔢 Top-k      : {args.top_k}\n")

    # ── Embed everything in one batch ─────────────────────────────────────────
    query_text = f"{QUERY_PREFIX}{args.query}"
    doc_texts = [f"{DOC_PREFIX}{item['code']} {item['text']}" for item in data]

    print("⏳ Embedding query + documents …")
    all_vecs = embed([query_text] + doc_texts)
    query_vec = np.array(all_vecs[0])
    doc_vecs = np.array(all_vecs[1:])
    print("✅ Embeddings ready.\n")

    # ── Plain top-k (for comparison) ──────────────────────────────────────────
    plain_scores = [cosine_similarity(query_vec, doc_vecs[i]) for i in range(len(data))]
    plain_top = sorted(range(len(data)), key=lambda i: plain_scores[i], reverse=True)[
        : args.top_k
    ]

    print("━" * 64)
    print(f"📊 PLAIN TOP-{args.top_k} (similarity only — may contain duplicates)")
    print("━" * 64)
    for rank, idx in enumerate(plain_top, 1):
        item = data[idx]
        print(
            f"  {rank}. [{plain_scores[idx]:.4f}] "
            f"{item['videoId']:12s}  {item['text'][:55]}…"
        )

    # ── MMR selection ─────────────────────────────────────────────────────────
    mmr_results = mmr_select(
        query_vec=query_vec,
        doc_vecs=doc_vecs,
        top_k=args.top_k,
        lambda_param=args.lambda_param,
    )

    print()
    print("━" * 64)
    print(f"🌈 MMR TOP-{args.top_k}  (λ={args.lambda_param} — relevant AND diverse)")
    print("━" * 64)
    print(f"  {'#':<3} {'MMR':>6} {'Rel':>6} {'Redund':>7}  {'videoId':<12}  Title")
    print(f"  {'─' * 3} {'─' * 6} {'─' * 6} {'─' * 7}  {'─' * 12}  {'─' * 40}")
    for rank, r in enumerate(mmr_results, 1):
        item = data[r["index"]]
        print(
            f"  {rank:<3} {r['mmr_score']:>6.4f} {r['relevance']:>6.4f} "
            f"{r['redundancy']:>7.4f}  {item['videoId']:<12}  {item['text'][:50]}…"
        )

    # ── Diversity gain summary ────────────────────────────────────────────────
    plain_set = set(plain_top)
    mmr_set = {r["index"] for r in mmr_results}
    swapped_in = mmr_set - plain_set
    swapped_out = plain_set - mmr_set

    if swapped_in:
        print()
        print("💡 Diversity gain — MMR swapped OUT these near-duplicates:")
        for idx in swapped_out:
            print(f"     ✂  {data[idx]['videoId']:12s} (sim={plain_scores[idx]:.4f})")
        print("   … and swapped IN these more-diverse results:")
        for idx in swapped_in:
            print(f"     ➕ {data[idx]['videoId']:12s} (sim={plain_scores[idx]:.4f})")
    else:
        print("\n✅ MMR and plain top-k agree — no near-duplicates were present.")


if __name__ == "__main__":
    main()
