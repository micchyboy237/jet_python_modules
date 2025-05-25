import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import util
from typing import List, Optional, Dict, TypedDict
from sklearn.cluster import AgglomerativeClustering
import json
import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from jet.file.utils import load_file, save_file
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
import time
from copy import deepcopy
from jet.logger import logger


class Header(TypedDict):
    node_id: str
    text: str
    doc_index: int
    header_level: int
    header: str
    parent_header: Optional[str]
    content: str
    chunk_index: Optional[int]
    token_count: Optional[int]
    source_url: Optional[str]


class PreprocessedText(TypedDict):
    text: str
    doc_index: int
    node_id: str
    header_level: int
    parent_header: Optional[str]
    header: str
    content: str
    chunk_index: Optional[int]
    token_count: Optional[int]
    source_url: Optional[str]


class MergedPreprocessedText(PreprocessedText):
    merged_count: Optional[int]
    merged_doc_chunk_indices: Optional[int]
    merged_doc_contents: List[str]
    merged_doc_headers: List[str]
    merged_docs: List[Header]
    tokens: int
    embed_score: Optional[float]
    rerank_score: Optional[float]


class SimilarityResult(TypedDict):
    node_id: str
    rank: int
    doc_index: int
    embed_score: float
    rerank_score: float
    score: float
    text: str
    tokens: int
    header_level: int
    parent_header: Optional[str]
    header: str
    content: str
    merged_docs: List[Header]


class EmbedResult(TypedDict):
    node_id: str
    rank: int
    doc_index: int
    score: float
    text: str
    tokens: int
    header_level: int
    parent_header: Optional[str]
    header: str
    content: str
    chunk_index: Optional[int]
    source_url: Optional[str]


class RerankResult(TypedDict):
    node_id: str
    rank: int
    doc_index: int
    score: float
    text: str
    tokens: int
    header_level: int
    parent_header: Optional[str]
    header: str
    content: str
    merged_docs: List[Header]
    embed_score: float
    rerank_score: float


class SearchResults(TypedDict):
    results: List[SimilarityResult]
    merge_results: List[MergedPreprocessedText]
    embed_results: List[EmbedResult]
    rerank_results: List[RerankResult]


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


def encode_chunk(chunk: List[str], model: SentenceTransformer, device: str) -> np.ndarray:
    return model.encode(chunk, batch_size=32, show_progress_bar=False, convert_to_tensor=True, device=device).cpu().numpy()


def embed_search(
    query: str,
    texts: List[Header],
    model_name: str = "all-MiniLM-L12-v2",
    *,
    device: str = get_device(),
    top_k: Optional[int] = None,
    num_threads: int = 4,
    min_header_level: int = 2,
    max_header_level: int = 6
) -> List[EmbedResult]:
    start_time = time.time()
    if not top_k:
        top_k = len(texts)
    logger.info(
        f"Starting embedding search for {len(texts)} texts, top_k={top_k}, device={device}, "
        f"min_header_level={min_header_level}, max_header_level={max_header_level}")
    filtered_texts = [
        t for t in texts
        if min_header_level <= t["header_level"] <= max_header_level
    ]
    logger.info(
        f"Filtered {len(texts) - len(filtered_texts)} texts outside header_level range "
        f"[{min_header_level}, {max_header_level}]. Remaining: {len(filtered_texts)}")
    if not filtered_texts:
        logger.warning(
            "No texts remain after header_level filtering, returning empty results")
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
    results = []
    for rank, idx in enumerate(top_k_indices, 1):
        tokens = len(model.tokenize([text_strings[idx]])["input_ids"][0])
        results.append({
            "node_id": filtered_texts[idx]["node_id"],
            "rank": rank,
            "doc_index": filtered_texts[idx]["doc_index"],
            "score": float(similarities[idx]),
            "text": filtered_texts[idx]["header"] + "\n" + filtered_texts[idx]["content"],
            "tokens": tokens,
            "header_level": filtered_texts[idx]["header_level"],
            "parent_header": filtered_texts[idx]["parent_header"],
            "header": filtered_texts[idx]["header"],
            "content": filtered_texts[idx]["content"],
            "chunk_index": filtered_texts[idx]["chunk_index"],
            "source_url": filtered_texts[idx]["source_url"],
        })
    logger.info(f"Embedding search returned {len(results)} results")
    if results:
        logger.debug(
            f"Top 3 candidates: {', '.join([f'{r['header'][:30]}... (score: {r['score']:.4f})' for r in results[:3]])}")
    logger.info(
        f"Embedding search completed in {time.time() - start_time:.2f} seconds")
    return results


def rerank_search(
    query: str,
    candidates: List[Dict],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
    device: str = get_device(),
    batch_size: int = 16
) -> List[RerankResult]:
    start_time = time.time()
    logger.info(
        f"Reranking {len(candidates)} candidates with batch_size={batch_size}")
    model = get_cross_encoder(model_name, device)
    pairs = [[query, candidate["text"]] for candidate in candidates]
    scores = model.predict(pairs, batch_size=batch_size)
    reranked = []
    for candidate, rerank_score in zip(candidates, scores):
        merged_docs = candidate.get("merged_docs", [{
            "node_id": candidate["node_id"],
            "text": candidate["text"],
            "doc_index": candidate["doc_index"],
            "header_level": candidate["header_level"],
            "header": candidate["header"],
            "parent_header": candidate["parent_header"],
            "content": candidate["content"],
            "chunk_index": candidate["chunk_index"],
            "token_count": candidate["tokens"],
            "source_url": candidate["source_url"]
        }])
        parent_header = candidate.get("parent_header")
        reranked.append({
            "node_id": candidate["node_id"],
            "rank": 0,
            "doc_index": candidate["doc_index"],
            "score": float(rerank_score),
            "text": candidate["text"],
            "tokens": candidate["tokens"],
            "header_level": candidate["header_level"],
            "parent_header": parent_header,
            "header": candidate["header"],
            "content": candidate["content"],
            "merged_docs": merged_docs,
            "embed_score": candidate.get("embed_score", 0.0),
            "rerank_score": float(rerank_score)
        })
    reranked = sorted(reranked, key=lambda x: x["score"], reverse=True)
    for rank, item in enumerate(reranked, 1):
        item["rank"] = rank
    logger.info(
        f"Reranking completed in {time.time() - start_time:.2f} seconds")
    return reranked


def merge_duplicate_texts_agglomerative(
    texts: List[EmbedResult],
    model_name: str = "all-MiniLM-L12-v2",
    device: str = "mps",
    similarity_threshold: float = 0.7,
    batch_size: int = 32
) -> List[MergedPreprocessedText]:
    logger.info(
        f"Deduplicating {len(texts)} texts based on headers with agglomerative clustering")
    start_time = time.time()
    if not texts:
        logger.warning(
            "No texts provided for deduplication, returning empty list")
        return []
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
    logger.debug(f"Similarity matrix shape: {similarities.shape}")
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
            "similarity_score": float(np.max(similarities[idx]))
        })
    deduplicated_texts: List[MergedPreprocessedText] = []
    for label, items in cluster_dict.items():
        if len(items) == 1:
            original = items[0]["original"]
            deduplicated_texts.append({
                "text": original["text"],
                "doc_index": original["doc_index"],
                "node_id": original["node_id"],
                "header_level": original["header_level"],
                "parent_header": original["parent_header"],
                "header": original["header"],
                "content": original["content"],
                "chunk_index": original["chunk_index"],
                "token_count": original["tokens"],
                "source_url": original["source_url"],
                "merged_count": 1,
                "merged_doc_chunk_indices": original["chunk_index"],
                "merged_doc_contents": [original["content"]],
                "merged_doc_headers": [original["header"]],
                "merged_docs": [original],
                "tokens": original["tokens"],
                "embed_score": original["score"],
                "rerank_score": None
            })
            logger.debug(
                f"Single text in cluster {label}: {original['header'][:30]}... (doc_index: {original['doc_index']})")
            continue
        items.sort(key=lambda x: x["similarity_score"], reverse=True)
        representative = items[0]["original"].copy()
        merged_content = "\n\n".join(
            item["original"]["content"] for item in items
        )
        merged_doc_chunk_indices = representative["chunk_index"]
        merged_doc_contents = [item["original"]["content"] for item in items]
        merged_doc_headers = [item["original"]["header"] for item in items]
        merged_docs = sorted(
            [item["original"] for item in items],
            key=lambda x: x["score"],
            reverse=True
        )
        total_tokens = sum(item["original"]["tokens"] for item in items)
        avg_embed_score = float(
            np.mean([item["original"]["score"] for item in items]))
        deduplicated_texts.append({
            "text": f"{representative['header']}\n{merged_content}",
            "doc_index": representative["doc_index"],
            "node_id": representative["node_id"],
            "header_level": representative["header_level"],
            "parent_header": representative["parent_header"],
            "header": representative["header"],
            "content": merged_content,
            "chunk_index": representative["chunk_index"],
            "token_count": total_tokens,
            "source_url": representative["source_url"],
            "merged_count": len(items),
            "merged_doc_chunk_indices": merged_doc_chunk_indices,
            "merged_doc_contents": merged_doc_contents,
            "merged_doc_headers": merged_doc_headers,
            "merged_docs": merged_docs,
            "tokens": total_tokens,
            "embed_score": avg_embed_score,
            "rerank_score": None
        })
        logger.debug(
            f"Merged {len(items)} texts for cluster {label}, header: {representative['header'][:30]}..., "
            f"doc_indices: {[item['original']['doc_index'] for item in items]}, new content length: {len(merged_content.split())} words")
    logger.info(
        f"Reduced {len(texts)} texts to {len(deduplicated_texts)} after header-based clustering. Time: {time.time() - start_time:.2f}s")
    return deduplicated_texts


def search_documents(
    query: str,
    headers: List[Header],
    model_name: str = "all-MiniLM-L12-v2",
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
    device: str = get_device(),
    top_k: int = 20,
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
) -> SearchResults:
    start_time = time.time()
    logger.info(
        f"Starting search with query: {query[:50]}..., {len(headers)} headers, device={device}")
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    if not headers:
        logger.warning("No headers provided, returning empty results")
        return {
            "results": [],
            "merge_results": [],
            "embed_results": [],
            "rerank_results": []
        }
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not 0 <= lambda_param <= 1:
        raise ValueError("lambda_param must be between 0 and 1")
    if not 0 <= parent_diversity_weight <= 1:
        raise ValueError("parent_diversity_weight must be between 0 and 1")
    if not 0 <= header_diversity_weight <= 1:
        raise ValueError("header_diversity_weight must be between 0 and 1")
    try:
        logger.info(f"Embedding search with {len(headers)} texts")
        embed_results = embed_search(
            query,
            headers,
            model_name,
            device=device,
            num_threads=num_threads,
            min_header_level=min_header_level,
            max_header_level=max_header_level,
        )
        logger.info(
            "Deduplicating embed results using agglomerative clustering")
        merged_texts = merge_duplicate_texts_agglomerative(
            embed_results,
            model_name=model_name,
            device=device,
            similarity_threshold=0.7,
            batch_size=32
        )
        logger.info(f"Reranking {len(merged_texts)} candidates")
        rerank_results_list = rerank_search(
            query, merged_texts, rerank_model, device, batch_size)
        results = [
            {
                "node_id": r["node_id"],
                "rank": 0,
                "doc_index": r["doc_index"],
                "embed_score": r["embed_score"],
                "rerank_score": r["rerank_score"],
                "score": (lambda_param * r["rerank_score"] + (1 - lambda_param) * r["embed_score"]),
                "text": r["text"],
                "tokens": r["tokens"],
                "header_level": r["header_level"],
                "parent_header": r["parent_header"],
                "header": r["header"],
                "content": r["content"],
                "merged_docs": r["merged_docs"]
            } for r in rerank_results_list[:top_k]
        ]
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        for rank, item in enumerate(results, 1):
            item["rank"] = rank
        logger.info(
            f"Search completed in {time.time() - start_time:.2f} seconds, returning {len(results)} results")
        return {
            "results": results,
            "merge_results": merged_texts,
            "embed_results": embed_results,
            "rerank_results": rerank_results_list
        }
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise RuntimeError(f"Search failed: {str(e)}") from e
