"""
Diversity-Preserving Dataset Sampling Pipeline
===============================================
Prevents overtraining on redundant data by removing duplicates at three
levels of granularity, following common industry/research practice
(SemDeDup [Abbas et al. 2023], NVIDIA NeMo Curator, SemHash):

  1. Exact dedup          -> hash-based, catches byte-identical records
  2. Near-lexical dedup   -> MinHash + LSH, catches similar wording
                              (paraphrase-light duplicates, boilerplate)
  3. Semantic dedup       -> embeddings + clustering + cosine threshold,
                              catches meaning-level duplicates even when
                              wording is totally different
  4. (Optional) MMR select -> final diverse top-k subset selection,
                              useful if you want a fixed-size, maximally
                              diverse sample rather than "dedup everything"

Embeddings are produced via the user's own local llama.cpp embedding server
(jet.adapters.llama_cpp.embed_utils.embed) instead of SentenceTransformer,
so nothing leaves the local machine and no HF model download is required.

Install:
    pip install datasketch scikit-learn numpy --break-system-packages

Usage:
    python diversity_sampling.py
"""

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from datasketch import MinHash, MinHashLSH

# Local embedding backend (llama.cpp server via OpenAI-compatible client)
from jet.adapters.llama_cpp.embed_utils import embed as llama_embed
from sklearn.cluster import KMeans


@dataclass
class DedupConfig:
    minhash_num_perm: int = 128
    minhash_threshold: float = 0.85  # Jaccard similarity -> near-lexical dupes
    embedding_model: str = None  # None -> use LLAMA_CPP_EMBED_MODEL env default
    embedding_batch_size: int = 32
    embedding_max_workers: int = 6
    n_clusters: int = 50  # tune relative to dataset size
    semantic_sim_threshold: float = 0.9  # cosine similarity -> semantic dupes
    random_seed: int = 42


# ---------------------------------------------------------------------------
# Step 1: Exact deduplication (byte-identical text)
# ---------------------------------------------------------------------------
def exact_dedup(texts: List[str]) -> List[int]:
    """Return indices of unique texts, keeping the first occurrence of each."""
    seen = set()
    keep = []
    for i, t in enumerate(texts):
        h = hashlib.sha256(t.strip().encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            keep.append(i)
    return keep


# ---------------------------------------------------------------------------
# Step 2: Near-lexical deduplication (MinHash + LSH)
# Catches items with overlapping wording/n-grams, e.g. minor edits,
# punctuation changes, boilerplate templates.
# ---------------------------------------------------------------------------
def _to_minhash(text: str, num_perm: int) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    for token in set(text.lower().split()):
        mh.update(token.encode("utf-8"))
    return mh


def near_lexical_dedup(
    texts: List[str], indices: List[int], cfg: DedupConfig
) -> List[int]:
    """Remove near-identical wording using MinHash LSH.
    Keeps the first-seen item within each near-duplicate group."""
    lsh = MinHashLSH(threshold=cfg.minhash_threshold, num_perm=cfg.minhash_num_perm)
    keep = []

    for idx in indices:
        mh = _to_minhash(texts[idx], cfg.minhash_num_perm)
        if not lsh.query(mh):  # no near-duplicate already indexed
            lsh.insert(str(idx), mh)
            keep.append(idx)
        # else: skip, it's a near-lexical duplicate of something already kept

    return keep


# ---------------------------------------------------------------------------
# Step 3: Embedding generation (for semantic dedup / MMR)
# Uses the user's local llama.cpp embedding server. embed_batch() already
# handles batching, parallel requests, and internal dedup of repeated
# strings -- we just need to L2-normalize the result ourselves, since
# semantic_dedup() / mmr_select() use plain dot products as cosine
# similarity and assume unit-length vectors.
# ---------------------------------------------------------------------------
def embed_texts(texts: List[str], indices: List[int], cfg: DedupConfig) -> np.ndarray:
    subset = [texts[i] for i in indices]

    kwargs = dict(
        return_format="numpy",
        max_workers=cfg.embedding_max_workers,
        batch_size=cfg.embedding_batch_size,
        show_progress=True,
    )
    if cfg.embedding_model:
        kwargs["model"] = cfg.embedding_model

    embeddings = llama_embed(subset, **kwargs)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # guard against zero vectors
    embeddings = embeddings / norms

    return embeddings


# ---------------------------------------------------------------------------
# Step 4: Semantic deduplication (SemDeDup-style)
# Cluster embeddings -> within each cluster, compute pairwise cosine
# similarity -> drop items too similar to something already kept.
# Kept items are visited in order of distance-from-centroid (most
# "distinctive" first), mirroring SemDeDup's approach of retaining the
# more novel point in each near-duplicate group.
# ---------------------------------------------------------------------------
def semantic_dedup(
    embeddings: np.ndarray, indices: List[int], cfg: DedupConfig
) -> List[int]:
    n = len(indices)
    n_clusters = min(cfg.n_clusters, n)  # can't exceed number of points

    kmeans = KMeans(n_clusters=n_clusters, random_state=cfg.random_seed, n_init="auto")
    cluster_ids = kmeans.fit_predict(embeddings)

    keep_mask = np.ones(n, dtype=bool)

    for c in range(n_clusters):
        member_positions = np.where(cluster_ids == c)[0]
        if len(member_positions) <= 1:
            continue

        cluster_embs = embeddings[member_positions]
        centroid = kmeans.cluster_centers_[c]
        dists_to_centroid = np.linalg.norm(cluster_embs - centroid, axis=1)
        order = np.argsort(-dists_to_centroid)  # farthest-from-centroid first

        kept_local = []
        for local_pos in order:
            emb = cluster_embs[local_pos]
            if kept_local:
                sims = (
                    cluster_embs[kept_local] @ emb
                )  # cosine sim (embeddings are normalized)
                if sims.max() >= cfg.semantic_sim_threshold:
                    keep_mask[member_positions[local_pos]] = False
                    continue
            kept_local.append(local_pos)

    return [indices[i] for i in range(n) if keep_mask[i]]


# ---------------------------------------------------------------------------
# Optional Step 5: MMR-based final subset selection
# Use this if, after dedup, you still want to pick a fixed-size k
# maximally-diverse sample (e.g. "give me the 500 most diverse rows").
# ---------------------------------------------------------------------------
def mmr_select(
    embeddings: np.ndarray,
    indices: List[int],
    k: int,
    relevance: np.ndarray = None,
    lambda_param: float = 0.5,
) -> List[int]:
    """
    Maximal Marginal Relevance selection.
    relevance: optional per-item relevance score (e.g. quality score).
               If None, all items are treated as equally relevant, so
               this becomes a pure diversity-maximizing sampler.
    """
    n = len(indices)
    k = min(k, n)
    if relevance is None:
        relevance = np.ones(n)

    selected: List[int] = []
    candidates = list(range(n))

    # seed with the single most relevant item
    first = int(np.argmax(relevance))
    selected.append(first)
    candidates.remove(first)

    while len(selected) < k and candidates:
        best_score, best_c = -np.inf, None
        for c in candidates:
            sim_to_selected = max(embeddings[c] @ embeddings[s] for s in selected)
            score = lambda_param * relevance[c] - (1 - lambda_param) * sim_to_selected
            if score > best_score:
                best_score, best_c = score, c
        selected.append(best_c)
        candidates.remove(best_c)

    return [indices[i] for i in selected]


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def diversity_preserving_dedup(
    texts: List[str], cfg: DedupConfig = DedupConfig()
) -> Tuple[List[int], Dict[str, int]]:
    """
    Runs the full 3-layer deduplication pipeline.
    Returns (kept_indices_into_original_texts, stats_dict).
    """
    stats = {"original": len(texts)}

    idx = exact_dedup(texts)
    stats["after_exact_dedup"] = len(idx)

    idx = near_lexical_dedup(texts, idx, cfg)
    stats["after_near_lexical_dedup"] = len(idx)

    if len(idx) > 1:
        embeddings = embed_texts(texts, idx, cfg)
        idx = semantic_dedup(embeddings, idx, cfg)
    stats["after_semantic_dedup"] = len(idx)

    return idx, stats


if __name__ == "__main__":
    sample_texts = [
        "The cat sat on the mat.",
        "The cat sat on the mat.",  # exact dup
        "The cat sat on the mat!!",  # near-lexical dup
        "A feline was resting on the rug.",  # semantic dup, different wording
        "Quantum computers use qubits instead of bits.",
        "Qubits are the fundamental unit of quantum computing.",  # semantic dup
        "The stock market fell sharply today.",
        "Paris is the capital of France.",
    ]

    cfg = DedupConfig(n_clusters=3, semantic_sim_threshold=0.75, minhash_threshold=0.8)
    kept_indices, stats = diversity_preserving_dedup(sample_texts, cfg)

    print("Stats:", stats)
    print("\nKept examples:")
    for i in kept_indices:
        print(f"  [{i}] {sample_texts[i]}")

    # --- Optional: pick a diverse subset of size k from what's left ---
    if len(kept_indices) > 1:
        embs = embed_texts(sample_texts, kept_indices, cfg)
        diverse_subset = mmr_select(embs, kept_indices, k=3, lambda_param=0.3)
        print("\nDiverse subset (MMR, k=3):")
        for i in diverse_subset:
            print(f"  [{i}] {sample_texts[i]}")
