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
from typing import Dict, List, Optional, Tuple

import numpy as np
from datasketch import MinHash, MinHashLSH
from jet.adapters.llama_cpp.embed_utils import embed
from sklearn.cluster import KMeans


@dataclass
class DedupConfig:
    minhash_num_perm: int = 128
    minhash_threshold: float = 0.85
    embedding_model: str = None
    embedding_batch_size: int = 32
    embedding_max_workers: int = 6
    n_clusters: int = 50
    semantic_sim_threshold: float = 0.9
    random_seed: int = 42


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
        if not lsh.query(mh):
            lsh.insert(str(idx), mh)
            keep.append(idx)
    return keep


def embed_texts(texts: List[str], indices: List[int], cfg: DedupConfig) -> np.ndarray:
    """Generate embeddings for the subset of texts defined by indices."""
    if not indices:
        return np.array([])

    subset = [texts[i] for i in indices]
    kwargs = dict(
        return_format="numpy",
        max_workers=cfg.embedding_max_workers,
        batch_size=cfg.embedding_batch_size,
        show_progress=True,
    )
    if cfg.embedding_model:
        kwargs["model"] = cfg.embedding_model

    embeddings = embed(subset, **kwargs)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms
    return embeddings


def semantic_dedup(
    embeddings: np.ndarray, indices: List[int], cfg: DedupConfig
) -> Tuple[List[int], np.ndarray]:
    """
    Performs semantic deduplication.

    Returns:
        Tuple of (kept_original_indices, kept_embeddings_aligned)
    """
    n = len(indices)
    if n == 0:
        return [], np.array([])

    n_clusters = min(cfg.n_clusters, n)
    kmeans = KMeans(n_clusters=n_clusters, random_state=cfg.random_seed, n_init="auto")
    cluster_ids = kmeans.fit_predict(embeddings)

    keep_mask = np.ones(n, dtype=bool)

    for c in range(n_clusters):
        member_positions = np.where(cluster_ids == c)[0]
        if len(member_positions) <= 1:
            continue

        cluster_embs = embeddings[member_positions]
        centroid = kmeans.cluster_centers_[c]

        # Sort by distance to centroid (furthest first)
        dists_to_centroid = np.linalg.norm(cluster_embs - centroid, axis=1)
        order = np.argsort(-dists_to_centroid)

        kept_local = []
        for local_pos in order:
            emb = cluster_embs[local_pos]
            if kept_local:
                sims = cluster_embs[kept_local] @ emb
                if sims.max() >= cfg.semantic_sim_threshold:
                    keep_mask[member_positions[local_pos]] = False
                    continue
            kept_local.append(local_pos)

    # Filter indices
    kept_indices = [indices[i] for i in range(n) if keep_mask[i]]
    # Filter embeddings to match kept indices
    kept_embeddings = embeddings[keep_mask]

    return kept_indices, kept_embeddings


def mmr_select(
    embeddings: np.ndarray,
    indices: List[int],
    k: int,
    relevance: np.ndarray = None,
    lambda_param: float = 0.5,
) -> List[int]:
    """
    Maximal Marginal Relevance selection.
    """
    n = len(indices)
    if n == 0:
        return []

    k = min(k, n)
    if relevance is None:
        relevance = np.ones(n)

    selected: List[int] = []
    candidates = list(range(n))

    # Start with the most relevant item
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

        if best_c is not None:
            selected.append(best_c)
            candidates.remove(best_c)

    return [indices[i] for i in selected]


def diversity_preserving_dedup(
    texts: List[str],
    cfg: DedupConfig = DedupConfig(),
    embeddings: Optional[np.ndarray] = None,
) -> Tuple[List[int], Dict[str, int], Optional[np.ndarray]]:
    """
    Runs the full 3-layer deduplication pipeline.

    Args:
        texts: List of input texts.
        cfg: Configuration for deduplication.
        embeddings: Optional pre-computed embeddings for ALL texts in the original list.
                    If provided, skips embedding generation.

    Returns:
        Tuple of (kept_indices_into_original_texts, stats_dict, kept_embeddings).
        kept_embeddings are normalized and aligned with kept_indices.
    """
    stats = {"original": len(texts)}

    # 1. Exact Dedup
    idx = exact_dedup(texts)
    stats["after_exact_dedup"] = len(idx)

    # 2. Near-Lexical Dedup
    idx = near_lexical_dedup(texts, idx, cfg)
    stats["after_near_lexical_dedup"] = len(idx)

    # 3. Semantic Dedup
    final_indices = idx
    final_embeddings = None

    if len(idx) > 1:
        if embeddings is not None:
            # Filter pre-computed embeddings for the current indices
            current_embs = embeddings[idx]
            # Normalize them
            norms = np.linalg.norm(current_embs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            current_embs = current_embs / norms
        else:
            # Generate embeddings only for the remaining candidates
            current_embs = embed_texts(texts, idx, cfg)

        # Perform semantic dedup
        final_indices, final_embeddings = semantic_dedup(current_embs, idx, cfg)

    elif len(idx) == 1:
        final_indices = idx
        if embeddings is not None:
            final_embeddings = embeddings[idx]
            norms = np.linalg.norm(final_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            final_embeddings = final_embeddings / norms
        else:
            final_embeddings = embed_texts(texts, idx, cfg)

    stats["after_semantic_dedup"] = len(final_indices)

    return final_indices, stats, final_embeddings


def get_diverse_subset(
    texts: List[str],
    k: int,
    cfg: DedupConfig = DedupConfig(),
    embeddings: Optional[np.ndarray] = None,
    relevance: Optional[np.ndarray] = None,
    lambda_param: float = 0.5,
) -> Tuple[List[int], List[str], Dict[str, int]]:
    """
    High-level function to perform deduplication and select a diverse subset.

    Args:
        texts: List of input texts.
        k: Number of diverse items to select.
        cfg: Deduplication configuration.
        embeddings: Optional pre-computed embeddings for all texts.
        relevance: Optional relevance scores for MMR.
        lambda_param: MMR trade-off parameter.

    Returns:
        Tuple of (selected_indices, selected_texts, stats).
    """
    # 1. Run Deduplication
    kept_indices, stats, deduped_embeddings = diversity_preserving_dedup(
        texts, cfg, embeddings
    )

    if not kept_indices:
        return [], [], stats

    # 2. Run MMR Selection
    # deduped_embeddings should already be available from the previous step
    if deduped_embeddings is None or len(deduped_embeddings) == 0:
        # Fallback if something went wrong, though unlikely
        deduped_embeddings = embed_texts(texts, kept_indices, cfg)

    diverse_indices = mmr_select(
        deduped_embeddings,
        kept_indices,
        k=k,
        relevance=relevance,
        lambda_param=lambda_param,
    )

    selected_texts = [texts[i] for i in diverse_indices]

    return diverse_indices, selected_texts, stats


if __name__ == "__main__":
    sample_texts = [
        "The cat sat on the mat.",
        "The cat sat on the mat.",
        "The cat sat on the mat!!",
        "A feline was resting on the rug.",
        "Quantum computers use qubits instead of bits.",
        "Qubits are the fundamental unit of quantum computing.",
        "The stock market fell sharply today.",
        "Paris is the capital of France.",
    ]

    cfg = DedupConfig(n_clusters=3, semantic_sim_threshold=0.75, minhash_threshold=0.8)

    print("--- Scenario 1: Automatic Embedding Generation ---")
    selected_indices, selected_texts, stats = get_diverse_subset(
        texts=sample_texts, k=3, cfg=cfg, lambda_param=0.3
    )

    print("Stats:", stats)
    print("\nSelected Diverse Subset:")
    for idx, text in zip(selected_indices, selected_texts):
        print(f"  [{idx}] {text}")

    print("\n--- Scenario 2: Using Pre-computed Embeddings ---")
    # Simulate pre-computed embeddings for all original texts
    print("Generating pre-computed embeddings for all texts...")
    all_embeddings = embed_texts(sample_texts, list(range(len(sample_texts))), cfg)

    selected_indices_2, selected_texts_2, stats_2 = get_diverse_subset(
        texts=sample_texts, k=3, cfg=cfg, embeddings=all_embeddings, lambda_param=0.3
    )

    print("Stats:", stats_2)
    print("\nSelected Diverse Subset (from pre-computed):")
    for idx, text in zip(selected_indices_2, selected_texts_2):
        print(f"  [{idx}] {text}")
