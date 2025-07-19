from collections import defaultdict
from typing import Dict, List, TypedDict
import numpy as np
from jet.logger import logger
from jet.vectors.semantic_search.search_types import Match
from jet.data.utils import generate_unique_id


class ChunkWithScore(TypedDict):
    id: str
    doc_id: str
    score: float
    header: str
    content: str
    parent_header: str
    matches: List[Match]
    metadata: Dict


def aggregate_doc_scores(
    chunks: List[ChunkWithScore],
    data_dict: Dict[str, dict],
    query_candidates: List[str]
) -> List[dict]:
    """Aggregate chunk scores into document scores, prioritizing longer n-gram matches and unique candidate matches."""
    doc_scores = defaultdict(list)
    for chunk in chunks:
        doc_id = chunk.get("doc_id")
        if doc_id is not None:
            doc_scores[doc_id].append(chunk)
            logger.debug(
                f"Added chunk {chunk['id']} to doc {doc_id}, query_scores={chunk['metadata'].get('query_scores', {})}")

    mapped_docs_with_scores = []
    for doc_id, chunks in doc_scores.items():
        doc = data_dict.get(doc_id, {})
        doc_text = doc.get("text", "")
        if not doc_text:
            logger.warning(f"Skipping doc {doc_id} due to empty text")
            continue
        doc_text_lower = doc_text.lower()
        chunk_offsets = []
        current_offset = 0
        for chunk in chunks:
            chunk_text = "\n".join([
                chunk.get("parent_header", "") or "",
                chunk["header"],
                chunk["content"]
            ]).strip()
            logger.debug(
                f"Doc {doc_id}: Searching for chunk text: {repr(chunk_text)}")
            chunk_start = doc_text_lower.find(
                chunk_text.lower(), current_offset)
            if chunk_start == -1:
                logger.warning(
                    f"Chunk {chunk['id']} not found in doc {doc_id} from offset {current_offset}")
                chunk_start = current_offset
            chunk_offsets.append(chunk_start)
            current_offset = chunk_start + len(chunk_text)
            logger.debug(
                f"Doc {doc_id}: Chunk {chunk['id']} start={chunk_start}, len={len(chunk_text)}, new_offset={current_offset}")

        logger.debug(f"Doc {doc_id}: Chunk offsets: {chunk_offsets}")
        all_matches = []
        for chunk, offset in zip(chunks, chunk_offsets):
            matches = chunk.get("matches", [])
            for match in matches:
                adjusted_match = Match(
                    text=match["text"],
                    start_idx=match["start_idx"] + offset,
                    end_idx=match["end_idx"] + offset
                )
                all_matches.append(adjusted_match)

        seen_matches = set()
        unique_matches = []
        for match in all_matches:
            match_tuple = (match["text"], match["start_idx"], match["end_idx"])
            if match_tuple not in seen_matches:
                unique_matches.append(match)
                seen_matches.add(match_tuple)
        unique_matches.sort(key=lambda m: (m["start_idx"], -m["end_idx"]))
        logger.debug(
            f"Doc {doc_id}: Adjusted and deduplicated matches: {[m for m in unique_matches]}")

        max_match_length = max(
            (m["end_idx"] - m["start_idx"] for m in unique_matches),
            default=0
        )
        # Boost based on longer n-grams with increased weight
        weight = 2.0 + 8 * np.log1p(max_match_length) / np.log1p(50)
        match_count = len(unique_matches)
        # Boost based on unique candidate terms matched
        unique_candidate_terms = len(
            set(m["text"].lower() for m in unique_matches))
        candidate_boost = 1.0 + 0.5 * \
            np.log1p(unique_candidate_terms) / np.log1p(len(query_candidates))
        logger.debug(
            f"Doc {doc_id}: Unique candidate terms: {unique_candidate_terms}, Candidate boost: {candidate_boost}")

        # Find the longest matched candidate term
        matched_candidates = [m["text"].lower() for m in unique_matches]
        longest_matched_candidate = max(
            matched_candidates, key=len, default="")
        # Apply exact match bonus only if the longest matched candidate equals the longest query candidate
        longest_query_candidate = max(query_candidates, key=len).lower()
        has_full_match = longest_matched_candidate == longest_query_candidate
        decay_factor = 1.0 if has_full_match else 1.0 / \
            (1.0 + 0.2 * max(0, match_count - 3))
        no_match_penalty = 0.3 if max_match_length == 0 else 1.0
        exact_match_bonus = 1.5 if has_full_match else 1.0

        chunk_scores = []
        for chunk in chunks:
            chunk_score = chunk["score"] * weight * decay_factor * \
                no_match_penalty * exact_match_bonus * candidate_boost
            chunk_scores.append(chunk_score)

        final_score = sum(chunk_scores) / len(chunks) if chunk_scores else 0.0
        logger.debug(
            f"Doc {doc_id}: Max match length: {max_match_length}, Decay factor: {decay_factor}, No match penalty: {no_match_penalty}, Exact match bonus: {exact_match_bonus}, Candidate boost: {candidate_boost}, Chunk scores: {chunk_scores}, Final score: {final_score}")

        if final_score == 0:
            logger.warning(f"Final score is 0 for doc {doc_id}")

        best_chunk = max(chunks, key=lambda c: c["score"])
        merged_metadata = {}
        doc_metadata = doc.get("metadata", {})
        chunk_metadata = best_chunk.get("metadata", {})
        merged_metadata.update(doc_metadata)
        merged_metadata.update(chunk_metadata)
        query_max_scores = defaultdict(float)
        for chunk in chunks:
            query_scores = chunk.get("metadata", {}).get("query_scores", {})
            for query_term, score in query_scores.items():
                query_max_scores[query_term] = max(
                    query_max_scores[query_term], score)
        merged_metadata["query_scores"] = dict(query_max_scores)

        mapped_doc = {
            "score": final_score,
            "id": doc_id,
            "text": doc_text,
            "posted_date": doc.get("posted_date", ""),
            "link": doc.get("link", ""),
            "num_tokens": doc.get("num_tokens", 0),
            "metadata": merged_metadata,
            "matches": unique_matches,
        }
        mapped_docs_with_scores.append(mapped_doc)

    mapped_docs_with_scores.sort(key=lambda d: (-d["score"], d.get("id", "")))
    for rank, doc in enumerate(mapped_docs_with_scores, 1):
        reordered_doc = {
            "rank": rank,
            **{k: v for k, v in doc.items()}
        }
        mapped_docs_with_scores[rank - 1] = reordered_doc

    return mapped_docs_with_scores
