import time
import logging
import random
from typing import List, Optional, TypedDict

import numpy as np
import torch


logger = logging.getLogger(__name__)


class TextSimilarityResult(TypedDict):
    id: str
    rank: int
    doc_index: int
    score: float
    text: str


def sort_by_mmr_diversity(
    candidates: List[str],
    num_results: int = 5,
    lambda_param: float = 0.5,
    text_diversity_weight: float = 0.4,
) -> List[TextSimilarityResult]:
    start_time = time.time()
    logger.info(f"Applying MMR diversity to select {num_results} results")

    # Convert string candidates to TextSimilarityResult format
    candidates_dict = [
        {
            "id": f"doc_{i}",
            "rank": 0,  # Will be set later
            "doc_index": i,
            "score": random.uniform(0, 1),  # Placeholder scoring
            "text": text
        }
        for i, text in enumerate(candidates)
    ]

    selected = []

    while len(selected) < num_results and candidates_dict:
        if not selected:
            best_candidate = candidates_dict.pop(0)
            selected.append(best_candidate)
            logger.debug(
                f"Selected first candidate: {best_candidate['text'][:30]}... (score: {best_candidate['score']:.4f})")
        else:
            mmr_scores = []
            selected_texts = [c["text"] for c in selected]

            for i, candidate in enumerate(candidates_dict):
                relevance = candidate["score"]
                text_penalty = text_diversity_weight if candidate["text"] in selected_texts else 0.0
                mmr_score = lambda_param * relevance - \
                    (1 - lambda_param) * text_penalty
                mmr_score = max(mmr_score, 0.0)
                mmr_scores.append(mmr_score)
                logger.debug(
                    f"Candidate {candidate['text'][:30]}...: mmr_score={mmr_score:.4f}, text_penalty={text_penalty:.2f}")

            best_idx = np.argmax(mmr_scores)
            best_candidate = candidates_dict.pop(best_idx)
            selected.append(best_candidate)
            logger.debug(
                f"Selected candidate {len(selected)}: {best_candidate['text'][:30]}... (score: {best_candidate['score']:.4f}, text_penalty: {text_penalty:.2f})")

    for rank, candidate in enumerate(selected, 1):
        candidate["rank"] = rank

    logger.info(
        f"MMR diversity selected {len(selected)} results: {', '.join([f'{r['text'][:30]}... (score: {r['score']:.4f})' for r in selected])}")
    logger.info(
        f"MMR diversity completed in {time.time() - start_time:.2f} seconds")
    return selected
