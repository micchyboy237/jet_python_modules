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

When no query is provided, the corpus embedding centroid is used as the
query vector — representing "what is this dataset mostly about?" — and
TF-IDF keywords are extracted purely for a human-readable label in the
output. This makes MMR's diversity effect more pronounced because all
documents are roughly equidistant from the centroid, so redundancy
penalties become the primary differentiator.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from jet.adapters.llama_cpp.embed_utils import embed
from sklearn.feature_extraction.text import TfidfVectorizer

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
            "final selection is both relevant AND non-redundant.\n\n"
            "When --query is omitted the corpus embedding centroid is used "
            "automatically, and top TF-IDF keywords are printed as a label."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        "--auto_keywords",
        type=int,
        default=6,
        metavar="N",
        help="Number of TF-IDF keywords to show when query is auto-derived.",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def extract_top_keywords(texts: list[str], n: int = 6) -> str:
    """
    Return the top-n TF-IDF keywords from a list of texts as a
    comma-separated string.  Used only for the auto-query display label —
    it does NOT affect the embedding or MMR computation.
    """
    vectorizer = TfidfVectorizer(
        max_features=n * 4,  # over-fetch then trim after sorting
        stop_words="english",
        ngram_range=(1, 2),  # single words + two-word phrases
        sublinear_tf=True,  # dampen very frequent terms
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    # Sum TF-IDF scores across all documents for each term
    scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
    top_indices = scores.argsort()[-n:][::-1]
    terms = vectorizer.get_feature_names_out()
    return ", ".join(terms[i] for i in top_indices)


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
    top_k = min(top_k, n)

    # Pre-compute relevance for all docs once — stays fixed throughout
    relevance_scores = np.array(
        [cosine_similarity(query_vec, doc_vecs[i]) for i in range(n)]
    )

    selected: list[int] = []
    results: list[dict] = []
    remaining = list(range(n))

    for _ in range(top_k):
        best_idx = None
        best_score = -np.inf
        best_rel = 0.0
        best_red = 0.0

        for idx in remaining:
            rel = relevance_scores[idx]

            # Redundancy = how similar this candidate is to our picks so far
            if selected:
                redundancy = max(
                    cosine_similarity(doc_vecs[idx], doc_vecs[s]) for s in selected
                )
            else:
                redundancy = 0.0  # first pick: no penalty

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

    # Texts used for embedding — include the prefix so the model understands
    # these are documents (not queries).
    doc_texts = [f"{DOC_PREFIX}{item['code']} {item['text']}" for item in data]
    # Texts used for TF-IDF keyword extraction — prefix stripped so that
    # "search" / "document" don't pollute the keyword vocabulary.
    clean_texts = [f"{item['code']} {item['text']}" for item in data]

    # ── Embed documents ───────────────────────────────────────────────────────
    print(f"📂 Loaded {len(data)} samples from : {samples_path.name}")
    print("⏳ Embedding documents …")
    doc_vecs = np.array(embed(doc_texts))
    print("✅ Document embeddings ready.\n")

    # ── Resolve query vector ──────────────────────────────────────────────────
    if args.query:
        # Explicit query supplied — embed it normally
        query_label = f'"{args.query}"'
        query_mode = "explicit"
        print("⏳ Embedding query …")
        query_vec = np.array(embed([f"{QUERY_PREFIX}{args.query}"])[0])
        print("✅ Query embedding ready.\n")
    else:
        # No query — use corpus centroid as the query vector.
        # TF-IDF keywords are extracted only to give the centroid a
        # human-readable name; they play no part in the MMR maths.
        query_vec = doc_vecs.mean(axis=0)
        query_mode = "auto (corpus centroid)"
        auto_kws = extract_top_keywords(clean_texts, n=args.auto_keywords)
        query_label = f"[auto: {auto_kws}]"

    lambda_desc = (
        "pure relevance"
        if args.lambda_param == 1.0
        else "pure diversity"
        if args.lambda_param == 0.0
        else "balanced"
    )

    print(f"🔍 Query mode : {query_mode}")
    print(f"🔍 Query      : {query_label}")
    print(f"🎛️  λ (lambda) : {args.lambda_param}  ({lambda_desc})")
    print(f"🔢 Top-k      : {args.top_k}\n")

    # ── Plain top-k (for before/after comparison) ─────────────────────────────
    plain_scores = [cosine_similarity(query_vec, doc_vecs[i]) for i in range(len(data))]
    plain_top = sorted(range(len(data)), key=lambda i: plain_scores[i], reverse=True)[
        : args.top_k
    ]

    print("━" * 68)
    print(f"📊 PLAIN TOP-{args.top_k}  (similarity only — may contain near-duplicates)")
    print("━" * 68)
    for rank, idx in enumerate(plain_top, 1):
        item = data[idx]
        print(
            f"  {rank}. [{plain_scores[idx]:.4f}]  "
            f"{item['videoId']:<12}  {item['text'][:55]}…"
        )

    # ── MMR selection ─────────────────────────────────────────────────────────
    mmr_results = mmr_select(
        query_vec=query_vec,
        doc_vecs=doc_vecs,
        top_k=args.top_k,
        lambda_param=args.lambda_param,
    )

    print()
    print("━" * 68)
    print(f"🌈 MMR TOP-{args.top_k}   (λ={args.lambda_param} — relevant AND diverse)")
    print("━" * 68)
    print(f"  {'#':<3} {'MMR':>6} {'Rel':>6} {'Redund':>7}  {'videoId':<12}  Title")
    print(f"  {'─' * 3} {'─' * 6} {'─' * 6} {'─' * 7}  {'─' * 12}  {'─' * 44}")
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

    print()
    if swapped_in:
        print("💡 Diversity gain — MMR swapped OUT these near-duplicates:")
        for idx in sorted(swapped_out, key=lambda i: plain_scores[i], reverse=True):
            print(f"     ✂  {data[idx]['videoId']:<12}  sim={plain_scores[idx]:.4f}")
        print("   … and swapped IN these more-diverse results:")
        for idx in sorted(swapped_in, key=lambda i: plain_scores[i], reverse=True):
            print(f"     ➕ {data[idx]['videoId']:<12}  sim={plain_scores[idx]:.4f}")
    else:
        print("✅ MMR and plain top-k agree — no near-duplicates were detected.")


if __name__ == "__main__":
    main()
