import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import util
from typing import List
from sklearn.cluster import AgglomerativeClustering
import json
import os
from typing import List, TypedDict, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from jet.file.utils import load_file, save_file
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
import time
from jet.logger import logger

# nltk.download('punkt')  # Download the Punkt tokenizer models


class Header(TypedDict):
    doc_index: int
    header_level: int
    header: str
    parent_header: str | None
    content: str
    chunk_index: int | None
    token_count: int | None
    source_url: str | None


class PreprocessedText(TypedDict):
    text: str
    doc_index: int
    id: str
    header_level: int
    parent_header: Optional[str]
    header: str
    content: str


class SimilarityResult(TypedDict):
    id: str
    rank: int
    doc_index: int
    score: float
    text: str
    tokens: int
    rerank_score: float
    diversity_score: float
    embedding: Optional[np.ndarray]
    header_level: int
    parent_header: Optional[str]
    header: str
    content: str


@lru_cache(maxsize=1)
def get_sentence_transformer(model_name: str, device: str) -> SentenceTransformer:
    logger.info(f"Loading SentenceTransformer model: {model_name} on {device}")
    return SentenceTransformer(model_name, device=device)


@lru_cache(maxsize=1)
def get_cross_encoder(model_name: str, device: str) -> CrossEncoder:
    logger.info(f"Loading CrossEncoder model: {model_name} on {device}")
    return CrossEncoder(model_name, device=device)


_device_cache = None


def get_device() -> str:
    global _device_cache
    if _device_cache is None:
        _device_cache = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Selected device: {_device_cache}")
    return _device_cache


def preprocess_texts(
    headers: List[Header],
    exclude_keywords: List[str] = [],
    min_header_words: int = 5,
    min_header_level: int = 2,
    parent_keyword: Optional[str] = None,
    min_content_words: int = 5
) -> List[PreprocessedText]:
    start_time = time.time()
    logger.info(
        f"Preprocessing {len(headers)} headers with min_header_words={min_header_words}, min_header_level={min_header_level}, min_content_words={min_content_words}, parent_keyword={parent_keyword}"
    )
    results = []
    keyword_excluded = 0
    short_header_excluded = 0
    short_content_excluded = 0
    level_excluded = 0
    parent_excluded = 0

    for i, header in enumerate(headers):
        content_words = header["content"].split()
        if len(content_words) < min_content_words:
            short_content_excluded += 1
            continue

        combined_text = f"{header['header']}\n{header['content']}"

        if any(keyword in header["header"].lower() for keyword in exclude_keywords) or \
           any(keyword in header["content"].lower() for keyword in exclude_keywords):
            keyword_excluded += 1
            continue
        if len(header["header"].split()) < min_header_words:
            short_header_excluded += 1
            continue
        if header["header_level"] < min_header_level:
            level_excluded += 1
            continue
        if parent_keyword and (not header["parent_header"] or parent_keyword.lower() not in header["parent_header"].lower()):
            parent_excluded += 1
            continue

        results.append({
            "text": combined_text,
            "doc_index": i,
            "id": f"doc_{i}",
            "header_level": header["header_level"],
            "chunk_index": header["chunk_index"],
            "token_count": header["token_count"],
            "source_url": header["source_url"],
            "parent_header": header["parent_header"],
            "header": header["header"],
            "content": header["content"]

        })

    logger.info(
        f"Preprocessed {len(results)} texts. Excluded: {keyword_excluded} (keywords), {short_header_excluded} (short headers), {short_content_excluded} (short content), {level_excluded} (header level), {parent_excluded} (parent keyword)"
    )
    if results:
        logger.debug(
            f"Sample preprocessed header: {json.dumps(results[0]['header'][:50])}..., content: {json.dumps(results[0]['content'][:50])}..."
        )
    logger.info(
        f"Preprocessing completed in {time.time() - start_time:.2f} seconds")
    return results


def encode_chunk(chunk: List[str], model: SentenceTransformer, device: str) -> np.ndarray:
    return model.encode(chunk, batch_size=32, show_progress_bar=False, convert_to_tensor=True, device=device).cpu().numpy()


def embed_search(
    query: str,
    texts: List[PreprocessedText],
    model_name: str = "all-mpnet-base-v2",
    device: str = get_device(),
    top_k: int = 20,
    num_threads: int = 4,
    max_header_level: int = 6
) -> List[SimilarityResult]:
    start_time = time.time()
    logger.info(
        f"Starting embedding search for {len(texts)} texts, top_k={top_k}, device={device}, max_header_level={max_header_level}")

    # Filter texts based on max_header_level
    filtered_texts = [
        t for t in texts if t["header_level"] <= max_header_level]
    logger.info(
        f"Filtered {len(texts) - len(filtered_texts)} texts with header_level > {max_header_level}. Remaining: {len(filtered_texts)}")

    if not filtered_texts:
        logger.warning(
            "No texts remain after max_header_level filtering, returning empty results")
        return []

    model = get_sentence_transformer(model_name, device)
    text_strings = [t["text"] for t in filtered_texts]
    query_embedding = model.encode(
        query, convert_to_tensor=True, device=device)
    chunk_size = 128
    similarities = []

    for i in range(0, len(text_strings), chunk_size):
        chunk = text_strings[i:i + chunk_size]
        chunk_embeddings = model.encode(
            chunk, batch_size=32, show_progress_bar=False, convert_to_tensor=True, device=device
        )
        chunk_similarities = util.cos_sim(query_embedding, chunk_embeddings)[
            0].cpu().numpy()
        similarities.append(chunk_similarities)
        del chunk_embeddings

    similarities = np.concatenate(similarities)
    top_k_indices = np.argsort(similarities)[
        ::-1][:min(top_k, len(similarities))]
    top_k_texts = [text_strings[idx] for idx in top_k_indices]
    top_k_embeddings = model.encode(
        top_k_texts, batch_size=32, show_progress_bar=False, convert_to_tensor=True, device=device
    ).cpu().numpy()

    results = []
    for rank, idx in enumerate(top_k_indices, 1):
        tokens = len(model.tokenize([text_strings[idx]])["input_ids"][0])
        results.append({
            "id": filtered_texts[idx]["id"],
            "rank": rank,
            "doc_index": filtered_texts[idx]["doc_index"],
            "chunk_index": filtered_texts[idx].get("chunk_index", None),
            "score": float(similarities[idx]),
            "text": text_strings[idx],
            "tokens": tokens,
            "rerank_score": 0.0,
            "diversity_score": 0.0,
            "embedding": top_k_embeddings[rank - 1],
            "header_level": filtered_texts[idx]["header_level"],
            "parent_header": filtered_texts[idx]["parent_header"],
            "header": filtered_texts[idx]["header"],
            "content": filtered_texts[idx]["content"],
            "source_url": filtered_texts[idx]["source_url"],
        })

    logger.info(f"Embedding search returned {len(results)} results")
    if results:
        logger.debug(
            f"Top 3 candidates: {', '.join([f'{r['header'][:30]}... (score: {r['score']:.4f})' for r in results[:3]])}")
    logger.info(
        f"Embedding search completed in {time.time() - start_time:.2f} seconds")
    return results


def rerank_results(
    query: str,
    candidates: List[SimilarityResult],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device: str = get_device(),
    batch_size: int = 16
) -> List[SimilarityResult]:
    start_time = time.time()
    logger.info(
        f"Reranking {len(candidates)} candidates with batch_size={batch_size}")
    model = get_cross_encoder(model_name, device)
    pairs = [[query, candidate["text"]] for candidate in candidates]
    scores = model.predict(pairs, batch_size=batch_size)
    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)
    reranked = sorted(
        candidates, key=lambda x: x["rerank_score"], reverse=True)
    for rank, candidate in enumerate(reranked, 1):
        candidate["rank"] = rank
    logger.info(
        f"Reranking completed. Top 3 reranked: {', '.join([f'{r['header'][:30]}... (rerank_score: {r['rerank_score']:.4f}, embedding_score: {r['score']:.4f})' for r in reranked[:3]])}")
    logger.info(
        f"Reranking completed in {time.time() - start_time:.2f} seconds")
    return reranked


def mmr_diversity(
    candidates: List[SimilarityResult],
    num_results: int = 5,
    lambda_param: float = 0.5,
    parent_diversity_weight: float = 0.4,
    header_diversity_weight: float = 0.3,
    device: str = get_device()
) -> List[SimilarityResult]:
    start_time = time.time()
    logger.info(f"Applying MMR diversity to select {num_results} results")
    selected = []
    candidate_embeddings = torch.tensor(
        np.array([c["embedding"] for c in candidates]), device=device)
    while len(selected) < num_results and candidates:
        if not selected:
            best_candidate = candidates.pop(0)
            best_candidate["diversity_score"] = best_candidate["rerank_score"]
            selected.append(best_candidate)
            logger.debug(
                f"Selected first candidate: {best_candidate['header'][:30]}... (rerank_score: {best_candidate['rerank_score']:.4f})")
        else:
            mmr_scores = []
            selected_embeddings = torch.tensor(
                np.array([c["embedding"] for c in selected]), device=device)
            selected_parents = [c["parent_header"] for c in selected]
            selected_contents = [c["content"] for c in selected]
            selected_headers = [c["header"] for c in selected]
            for i, candidate in enumerate(candidates):
                relevance = candidate["rerank_score"]
                similarity = float(np.max(
                    util.cos_sim(
                        candidate_embeddings[i:i+1],
                        selected_embeddings
                    ).cpu().numpy()
                ))
                parent_penalty = parent_diversity_weight if candidate[
                    "parent_header"] in selected_parents else 0.0
                content_penalty = parent_diversity_weight if any(
                    candidate["content"] == content for content in selected_contents
                ) else 0.0
                header_penalty = 0.0
                for selected_header in selected_headers:
                    similarity_ratio = SequenceMatcher(
                        None, candidate["header"].lower(), selected_header.lower()).ratio()
                    if similarity_ratio > 0.6:
                        header_penalty = header_diversity_weight
                        break
                mmr_score = lambda_param * relevance - \
                    (1 - lambda_param) * (similarity +
                                          parent_penalty + content_penalty + header_penalty)
                mmr_score = max(mmr_score, 0.0)
                mmr_scores.append(mmr_score)
                candidate["diversity_score"] = float(mmr_score)
                logger.debug(
                    f"Candidate {candidate['header'][:30]}...: mmr_score={mmr_score:.4f}, parent_penalty={parent_penalty:.2f}, header_penalty={header_penalty:.2f}, content_penalty={content_penalty:.2f}")
            best_idx = np.argmax(mmr_scores)
            best_candidate = candidates.pop(best_idx)
            selected.append(best_candidate)
            logger.debug(
                f"Selected candidate {len(selected)}: {best_candidate['header'][:30]}... (diversity_score: {best_candidate['diversity_score']:.4f}, parent_penalty: {parent_penalty:.2f}, header_penalty: {header_penalty:.2f}, content_penalty: {content_penalty:.2f})")
            del selected_embeddings
    del candidate_embeddings
    for rank, candidate in enumerate(selected, 1):
        candidate["rank"] = rank
    logger.info(
        f"MMR diversity selected {len(selected)} results: {', '.join([f'{r['header'][:30]}... (diversity_score: {r['diversity_score']:.4f})' for r in selected])}")
    logger.info(
        f"MMR diversity completed in {time.time() - start_time:.2f} seconds")
    return selected


def merge_duplicate_texts_agglomerative(
    texts: List[PreprocessedText],
    model_name: str = "all-MiniLM-L12-v2",
    device: str = "mps",
    similarity_threshold: float = 0.7,
    batch_size: int = 32
) -> List[PreprocessedText]:
    logger.info(
        f"Deduplicating {len(texts)} texts based on headers with agglomerative clustering")
    start_time = time.time()
    headers = [t["header"] for t in texts]
    model = get_sentence_transformer(model_name, device)
    embeddings = model.encode(
        headers,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_tensor=True,
        device=device
    ).cpu().numpy()
    similarities = util.cos_sim(embeddings, embeddings).cpu().numpy()
    distances = 1 - similarities
    logger.debug(f"Similarity matrix:\n{similarities}")
    logger.debug(f"Distance matrix:\n{distances}")
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1 - similarity_threshold,
        metric="precomputed",
        linkage="average"
    )
    labels = clustering.fit_predict(distances)
    logger.debug(f"Cluster labels: {labels}")
    cluster_dict = {}
    for idx, (label, text) in enumerate(zip(labels, texts)):
        if label not in cluster_dict:
            cluster_dict[label] = []
        cluster_dict[label].append({
            "original": text,
            "embedding": embeddings[idx],
            "similarity_score": np.max(similarities[idx])
        })
    deduplicated_texts = []
    for label, items in cluster_dict.items():
        if len(items) == 1:
            deduplicated_texts.append(items[0]["original"])
            logger.debug(
                f"Single text in cluster {label}: {items[0]['original']['header'][:30]}...")
            continue
        items.sort(key=lambda x: x["original"]["doc_index"])
        representative = items[0]["original"].copy()
        merged_content = "\n\n".join(
            item["original"]["content"] for item in items
        )
        representative["content"] = merged_content[:500]
        deduplicated_texts.append(representative)
        logger.debug(
            f"Merged {len(items)} texts for cluster {label}, header: {representative['header'][:30]}..., new content length: {len(merged_content.split())} words")
    logger.info(
        f"Reduced {len(texts)} texts to {len(deduplicated_texts)} after header-based clustering. Time: {time.time() - start_time:.2f}s")
    return deduplicated_texts


def search_documents(
    query: str,
    headers: List[Header],
    model_name: str = "all-mpnet-base-v2",
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device: str = get_device(),
    top_k: int = 20,
    num_results: int = 5,
    lambda_param: float = 0.5,
    batch_size: int = 16,
    num_threads: int = 4,
    exclude_keywords: List[str] = [],
    min_header_words: int = 5,
    min_header_level: int = 2,
    max_header_level: int = 6,
    parent_keyword: Optional[str] = None,
    parent_diversity_weight: float = 0.4,
    header_diversity_weight: float = 0.3,
    min_content_words: int = 5
) -> List[SimilarityResult]:
    start_time = time.time()
    logger.info(
        f"Starting search with query: {query[:50]}..., {len(headers)} headers, device={device}")
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    if not headers:
        logger.warning("No headers provided, returning empty results")
        return []
    if top_k < 1 or num_results < 1:
        raise ValueError("top_k and num_results must be positive")
    if not 0 <= lambda_param <= 1:
        raise ValueError("lambda_param must be between 0 and 1")
    if not 0 <= parent_diversity_weight <= 1:
        raise ValueError("parent_diversity_weight must be between 0 and 1")
    if not 0 <= header_diversity_weight <= 1:
        raise ValueError("header_diversity_weight must be between 0 and 1")
    try:
        logger.info("Preprocessing texts")
        texts = preprocess_texts(headers, exclude_keywords, min_header_words,
                                 min_header_level, parent_keyword, min_content_words)
        if not texts:
            logger.warning(
                "No texts after preprocessing, returning empty results")
            return []
        logger.info("Deduplicating texts using agglomerative clustering")
        texts = merge_duplicate_texts_agglomerative(
            texts,
            model_name=model_name,
            device=device,
            similarity_threshold=0.7,
            batch_size=32
        )
        logger.info(f"Embedding search with {len(texts)} texts")
        candidates = embed_search(
            query, texts, model_name, device, top_k, num_threads, max_header_level=max_header_level)
        logger.info(f"Reranking {len(candidates)} candidates")
        reranked = rerank_results(
            query, candidates, rerank_model, device, batch_size)
        logger.info(f"Applying MMR diversity to select {num_results} results")
        diverse_results = mmr_diversity(
            reranked, num_results, lambda_param, parent_diversity_weight, header_diversity_weight, device)
        diverse_results = sorted(
            diverse_results, key=lambda x: x["score"], reverse=True)
        logger.info(
            f"Search completed in {time.time() - start_time:.2f} seconds, returning {len(diverse_results)} results")
        return diverse_results
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise RuntimeError(f"Search failed: {str(e)}") from e
