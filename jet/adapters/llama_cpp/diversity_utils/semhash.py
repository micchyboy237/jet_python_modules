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

Embedding contract
------------------
`diversify_results` and `shuffle_and_diversify` operate on **precomputed
embeddings**. They never re-embed. Callers must pass a 2-D numpy array
aligned 1:1 with `results` (and, for the shuffle path, with the *full*
candidate pool before sampling). This is enforced by shape/id checks
below, so a misalignment fails loudly instead of silently degrading to
garbage MMR (see the previous `get_embeddings_fn` fallback, which used
to re-encode candidate *IDs* through the llama.cpp server).

If you genuinely want to embed raw documents, use `embed(...)` from
`embed_utils` first and pass the result in.
"""

import logging
import time
from typing import TypedDict

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


class LlamaCppSemHashEncoder:
    """
    Semhash-compatible Encoder (see semhash.utils.Encoder protocol — only
    needs `.encode()`), backed by our local llama.cpp embedding server.

    This encoder is *only* invoked by SemHash during the MMR diversity
    re-ranking step (self_find_representative -> _diversify), and even then
    only if `SemHash.from_records` was used instead of `SemHash.from_embeddings`.
    Since `diversify_results` always uses `from_embeddings`, this class is
    effectively a no-op inside this module — it exists solely so the SemHash
    constructor has a non-None `model` (required by `from_embeddings`).

    Do **not** call `encode()` on candidate IDs; it expects documents.
    """

    def encode(self, inputs: str | list[str], **kwargs) -> np.ndarray:
        """
        Encode a string or list of strings into embeddings.

        :param inputs: A string or a list of strings (documents) to encode.
        :param **kwargs: Ignored. Kept for Encoder protocol compatibility.
        :return: Embeddings as a numpy array of shape (n_inputs, embedding_dim).
        """
        texts = [inputs] if isinstance(inputs, str) else list(inputs)
        logger.debug("🧠 [LlamaCppSemHashEncoder] Encoding %d document(s)", len(texts))
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
            "⚠️ [SearchDiversity] compute_fetch_k capped: wanted %d "
            "candidates (top_k=%d, diversity=%.2f, multiplier=%dx) but "
            "capped to %d. MMR is working with a smaller pool than "
            "intended for this top_k.",
            desired,
            top_k,
            diversity,
            multiplier,
            FETCH_K_CAP,
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
            "⚠️ [SearchDiversity] compute_shuffle_fetch_k capped: wanted "
            "%d candidates (top_k=%d) but capped to %d. Shuffle variety "
            "is reduced for this top_k.",
            desired,
            top_k,
            SHUFFLE_FETCH_K_CAP,
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
        "🔀 [SearchDiversity] Sampled %d/%d candidates (%d guaranteed "
        "top-scored + %d weighted random, seed=%d, relevance_bias=%.2f)",
        sample_size,
        len(results),
        guaranteed_top,
        remaining_sample_size,
        seed,
        relevance_bias,
    )
    return guaranteed + sampled_rest


def _validate_embedding_alignment(
    embeddings: np.ndarray,
    results: list[SearchResult],
    *,
    caller: str,
) -> bool:
    """
    Ensure `embeddings` is a 2-D array with one row per result.

    Returns True when aligned; returns False (and logs an error) otherwise.
    Callers should short-circuit to "no diversification" on False.
    """
    if not isinstance(embeddings, np.ndarray):
        logger.error(
            "⚠️ [SearchDiversity] %s: embeddings must be a numpy array, got %s",
            caller,
            type(embeddings).__name__,
        )
        return False
    if embeddings.ndim != 2:
        logger.error(
            "⚠️ [SearchDiversity] %s: embeddings must be 2-D, got shape %s",
            caller,
            embeddings.shape,
        )
        return False
    if embeddings.shape[0] != len(results):
        logger.error(
            "⚠️ [SearchDiversity] %s: embeddings/results mismatch (%d vs %d). "
            "Passing IDs instead of documents? Re-embedding is disabled by "
            "design — callers must supply precomputed embeddings aligned 1:1 "
            "with `results`.",
            caller,
            embeddings.shape[0],
            len(results),
        )
        return False
    return True


def diversify_results(
    results: list[SearchResult],
    top_k: int,
    diversity: float,
    embeddings: np.ndarray,
) -> list[SearchResult]:
    """
    Select a diverse subset of `results` using SemHash's MMR-based
    diversification (pyversity under the hood).

    `embeddings` MUST be precomputed and aligned 1:1 with `results`. This
    function never re-embeds: the whole point of the `from_embeddings`
    path is to reuse vectors that are already sitting in Postgres.

    :param results: Candidate search results.
    :param top_k: Number of diverse representatives to select.
    :param diversity: Trade-off between diversity (1.0) and relevance (0.0).
    :param embeddings: Precomputed embeddings, shape (len(results), dim).
    :return: Diversified subset of `results`.
    :raises ValueError: If `embeddings` is misaligned with `results`.
    """
    if len(results) <= top_k:
        return results

    if not _validate_embedding_alignment(
        embeddings, results, caller="diversify_results"
    ):
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
        "🎨 [SearchDiversity] Diversified %d candidates -> %d results "
        "(diversity=%.2f) in %.2fms",
        len(results),
        len(diversified.selected),
        diversity,
        elapsed,
    )
    return diversified.selected


def shuffle_and_diversify(
    results: list[SearchResult],
    top_k: int,
    diversity: float,
    seed: int,
    relevance_bias: float = BALANCED_RELEVANCE_BIAS,
    embeddings: np.ndarray = None,
) -> list[SearchResult]:
    """
    Shuffle entry point: weighted-sample the overfetched candidate pool
    down to a workable size, then run the normal MMR diversify step on
    that sample.

    `embeddings` MUST be precomputed and aligned 1:1 with the **full**
    `results` list (sampling happens first, then we slice the embeddings
    to match the sample). This function never re-embeds.

    :param results: The overfetched, score-filtered candidate pool.
    :param top_k: Number of final results to return.
    :param diversity: Trade-off between diversity (1.0) and relevance (0.0).
    :param seed: Random seed for the shuffle.
    :param relevance_bias: Exponent controlling relevance-vs-variety balance.
    :param embeddings: Precomputed embeddings for the entire `results` pool.
    :return: A diverse top_k subset, varying by `seed`.
    :raises ValueError: If `embeddings` is missing or misaligned with `results`.
    """
    if embeddings is None:
        raise ValueError(
            "shuffle_and_diversify requires precomputed `embeddings` aligned "
            "with the FULL `results` list (re-embedding is disabled by design)."
        )

    if not _validate_embedding_alignment(
        embeddings, results, caller="shuffle_and_diversify"
    ):
        return results[:top_k]

    sample_size = min(len(results), compute_fetch_k(top_k, diversity))
    sampled = sample_candidate_pool(
        results, sample_size=sample_size, seed=seed, relevance_bias=relevance_bias
    )

    # Slice embeddings to match the sampled subset. `sample_candidate_pool`
    # only ever returns items drawn from `results`, so every sampled id
    # must exist in the id→index map; if not, fail loudly rather than
    # silently producing misaligned embeddings.
    id_to_idx = {r["id"]: i for i, r in enumerate(results)}
    missing = [r["id"] for r in sampled if r["id"] not in id_to_idx]
    if missing:
        logger.error(
            "⚠️ [SearchDiversity] shuffle_and_diversify: %d sampled IDs not "
            "present in the embeddings map (e.g. %s). This indicates the "
            "sampler and `results` disagree — skipping diversification.",
            len(missing),
            missing[:3],
        )
        return results[:top_k]

    sampled_indices = [id_to_idx[r["id"]] for r in sampled]
    sampled_embeddings = embeddings[sampled_indices]

    return diversify_results(
        sampled,
        top_k=top_k,
        diversity=diversity,
        embeddings=sampled_embeddings,
    )
