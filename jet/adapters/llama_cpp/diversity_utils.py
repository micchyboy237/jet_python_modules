"""Diversity-aware result selection for semantic search.
Isolated here so chroma_service.py stays focused on CRUD/search plumbing.
This module owns:
  - the SemHash-compatible encoder adapter for our llama.cpp server
  - the MMR-based diversification step applied over an already-fetched
    candidate pool (embeddings are reused from ChromaDB, not recomputed)
  - the "shuffle" sampling step: a weighted random resample of the
    candidate pool, used before diversify() to vary which results surface
    across repeated calls for the same query, while staying biased toward
    relevance (see sample_candidate_pool)
"""

import logging
import time
from typing import Callable, Optional, TypedDict

import numpy as np
from jet.adapters.llama_cpp.embed_utils import embed
from semhash import SemHash

logger = logging.getLogger(__name__)

BALANCED_RELEVANCE_BIAS = 1.0
MIN_SAMPLE_WEIGHT = 0.05
FETCH_K_CAP = 300
SHUFFLE_FETCH_K_CAP = 400


class SearchResult(TypedDict):
    """Represents a single search result candidate."""

    id: str
    score: float
    document: str


# Type alias for the embedding fetcher function
EmbeddingFn = Callable[[list[str]], Optional[np.ndarray]]


class LlamaCppSemHashEncoder:
    """
    Semhash-compatible Encoder (see semhash.utils.Encoder protocol — only
    needs `.encode()`), backed by our local llama.cpp embedding server.
    Only invoked by SemHash during the MMR diversity re-ranking step
    (self_find_representative -> _diversify), since the initial index is
    built from embeddings already stored in ChromaDB via
    SemHash.from_embeddings() — no full re-embedding of the candidate pool.
    """

    def encode(self, inputs: str | list[str], **kwargs) -> np.ndarray:
        """
        Encode a string or list of strings into embeddings.
        :param inputs: A string or a list of strings to encode.
        :param **kwargs: Ignored. Kept for Encoder protocol compatibility.
        :return: Embeddings as a numpy array of shape (n_inputs, embedding_dim).
        """
        texts = [inputs] if isinstance(inputs, str) else list(inputs)
        logger.debug(
            f"🧠 [LlamaCppSemHashEncoder] Re-encoding {len(texts)} candidate(s)"
        )
        return embed(texts, return_format="numpy", show_progress=False)


def compute_fetch_k(top_k: int, diversity: float) -> int:
    """
    Compute how many candidates to overfetch before diversifying (normal,
    non-shuffle path).
    """
    if diversity <= 0:
        return top_k
    multiplier = 3 + round(diversity * 3)
    desired = top_k * multiplier
    fetch_k = min(max(desired, top_k + 20), FETCH_K_CAP)
    if desired > FETCH_K_CAP:
        logger.warning(
            f"⚠️ [SearchDiversity] compute_fetch_k capped: wanted {desired} "
            f"candidates (top_k={top_k}, diversity={diversity}, "
            f"multiplier={multiplier}x) but capped to {FETCH_K_CAP}. "
            f"MMR is working with a smaller pool than intended for this top_k."
        )
    return fetch_k


def compute_shuffle_fetch_k(top_k: int) -> int:
    """
    Compute how many candidates to overfetch for a SHUFFLE request.
    """
    desired = top_k * 10
    fetch_k = min(max(desired, top_k + 60), SHUFFLE_FETCH_K_CAP)
    if desired > SHUFFLE_FETCH_K_CAP:
        logger.warning(
            f"⚠️ [SearchDiversity] compute_shuffle_fetch_k capped: wanted "
            f"{desired} candidates (top_k={top_k}) but capped to "
            f"{SHUFFLE_FETCH_K_CAP}. Shuffle variety is reduced for this top_k."
        )
    return fetch_k


def sample_candidate_pool(
    results: list[SearchResult],
    sample_size: int,
    seed: int,
    relevance_bias: float = BALANCED_RELEVANCE_BIAS,
    guaranteed_top: int = 1,
) -> list[SearchResult]:
    """
    Weighted random sample (without replacement) of `results`, biased
    toward higher relevance scores.
    """
    if sample_size >= len(results):
        return results

    guaranteed_top = min(guaranteed_top, sample_size, len(results))
    sorted_by_score = sorted(results, key=lambda r: r["score"], reverse=True)
    guaranteed = sorted_by_score[:guaranteed_top]
    guaranteed_ids = {r["id"] for r in guaranteed}

    remaining_pool = [r for r in results if r["id"] not in guaranteed_ids]
    remaining_sample_size = sample_size - guaranteed_top

    if remaining_sample_size <= 0 or not remaining_pool:
        return guaranteed

    scores = np.asarray([r["score"] for r in remaining_pool], dtype=np.float64)
    score_range = scores.max() - scores.min()
    normalized = (
        (scores - scores.min()) / score_range
        if score_range > 0
        else np.ones_like(scores)
    )
    weights = np.power(normalized, relevance_bias) + MIN_SAMPLE_WEIGHT
    weights = weights / weights.sum()

    rng = np.random.default_rng(seed)
    indices = rng.choice(
        len(remaining_pool), size=remaining_sample_size, replace=False, p=weights
    )
    sampled_rest = [remaining_pool[i] for i in indices]

    logger.debug(
        f"🔀 [SearchDiversity] Sampled {sample_size}/{len(results)} candidates "
        f"({guaranteed_top} guaranteed top-scored + {remaining_sample_size} "
        f"weighted random, seed={seed}, relevance_bias={relevance_bias})"
    )
    return guaranteed + sampled_rest


def diversify_results(
    results: list[SearchResult],
    top_k: int,
    diversity: float,
    get_embeddings_fn: Optional[EmbeddingFn] = None,
    embeddings: Optional[np.ndarray] = None,
) -> list[SearchResult]:
    """
    Select a diverse subset of `results` using SemHash's MMR-based
    diversification (pyversity under the hood).

    :param results: Candidate search results.
    :param top_k: Number of diverse representatives to select.
    :param diversity: Trade-off between diversity (1.0) and relevance (0.0).
    :param get_embeddings_fn: Optional callable to fetch embeddings if not provided directly.
    :param embeddings: Optional pre-computed embeddings. If provided, skips get_embeddings_fn.
    :return: Diversified subset of `results`.
    """
    if len(results) <= top_k:
        return results

    # Resolve embeddings
    if embeddings is None:
        if get_embeddings_fn is None:
            raise ValueError(
                "Either 'embeddings' or 'get_embeddings_fn' must be provided."
            )

        ids = [r["id"] for r in results]
        embeddings = get_embeddings_fn(ids)

        if embeddings is None or len(embeddings) != len(results):
            logger.warning(
                f"⚠️ [SearchDiversity] Could not retrieve aligned embeddings for "
                f"{len(ids)} candidates — skipping diversification, returning "
                f"top_k by relevance"
            )
            return results[:top_k]

    start_time = time.time()
    semhash = SemHash.from_embeddings(
        embeddings=embeddings,
        records=results,
        model=LlamaCppSemHashEncoder(),
        columns=["document"],
    )
    diversified = semhash.self_find_representative(
        diversity=diversity,
        selection_size=top_k,
    )
    elapsed = (time.time() - start_time) * 1000
    logger.info(
        f"🎨 [SearchDiversity] Diversified {len(results)} candidates -> "
        f"{len(diversified.selected)} results (diversity={diversity}) "
        f"in {elapsed:.2f}ms"
    )
    return diversified.selected


def shuffle_and_diversify(
    results: list[SearchResult],
    top_k: int,
    diversity: float,
    seed: int,
    get_embeddings_fn: Optional[EmbeddingFn] = None,
    relevance_bias: float = BALANCED_RELEVANCE_BIAS,
    embeddings: Optional[np.ndarray] = None,
) -> list[SearchResult]:
    """
    Shuffle entry point: weighted-sample the overfetched candidate pool
    down to a workable size, then run the normal MMR diversify step on
    that sample.

    :param results: The overfetched, score-filtered candidate pool.
    :param top_k: Number of final results to return.
    :param diversity: Trade-off between diversity (1.0) and relevance (0.0).
    :param seed: Random seed for the shuffle.
    :param get_embeddings_fn: Optional callable to fetch embeddings.
    :param relevance_bias: Exponent controlling relevance-vs-variety balance.
    :param embeddings: Optional pre-computed embeddings for the entire 'results' pool.
        Note: If provided, these embeddings must correspond to the FULL 'results' list,
        as sampling happens first, and we slice these embeddings to match the sample.
    :return: A diverse top_k subset, varying by `seed`.
    """
    sample_size = min(len(results), compute_fetch_k(top_k, diversity))
    sampled = sample_candidate_pool(
        results, sample_size=sample_size, seed=seed, relevance_bias=relevance_bias
    )

    # If pre-computed embeddings were provided for the full pool,
    # we need to extract the embeddings for the sampled items.
    sampled_embeddings = None
    if embeddings is not None:
        # Create a map of ID to index in the original results/embeddings array
        id_to_idx = {r["id"]: i for i, r in enumerate(results)}
        sampled_indices = [id_to_idx[r["id"]] for r in sampled]
        sampled_embeddings = embeddings[sampled_indices]

    return diversify_results(
        sampled,
        top_k=top_k,
        diversity=diversity,
        get_embeddings_fn=get_embeddings_fn,
        embeddings=sampled_embeddings,
    )
