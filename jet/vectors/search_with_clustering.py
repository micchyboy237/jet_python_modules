from sentence_transformers import util
from typing import List, Optional, Dict, TypedDict
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from functools import lru_cache
import time
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


class MergeInfo(TypedDict):
    min_tokens: int
    max_tokens: int
    avg_tokens: float


class MergedPreprocessedText(PreprocessedText):
    merged_count: Optional[int]
    merged_doc_chunk_indices: Optional[int]
    merged_doc_contents: List[str]
    merged_doc_headers: List[str]
    merged_docs: List[Header]
    tokens: int
    embed_score: Optional[float]
    rerank_score: Optional[float]
    merge_info: Optional[MergeInfo]


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
    embed_score: float
    rerank_score: float
    chunk_index: Optional[int]
    source_url: Optional[str]


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
    merge_info: Optional[MergeInfo]
    chunk_index: Optional[int]
    source_url: Optional[str]


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
    text_strings = [
        f"{t["parent_header"]}\n{t["text"]}" for t in filtered_texts]
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
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device: str = get_device(),
    batch_size: int = 16
) -> List[RerankResult]:
    start_time = time.time()
    logger.info(
        f"Reranking {len(candidates)} candidates with batch_size={batch_size}")
    if not candidates:
        logger.warning(
            "No candidates provided for reranking, returning empty list")
        return []
    model = get_cross_encoder(model_name, device)
    pairs = [[query, candidate["text"]] for candidate in candidates]
    scores = model.predict(pairs, batch_size=batch_size)
    reranked = []
    for candidate, rerank_score in zip(candidates, scores):
        embed_score = candidate.get("embed_score", candidate.get("score", 0.0))
        reranked.append({
            "node_id": candidate["node_id"],
            "rank": 0,
            "doc_index": candidate["doc_index"],
            "score": float(rerank_score),
            "text": candidate["text"],
            "tokens": candidate["tokens"],
            "header_level": candidate["header_level"],
            "parent_header": candidate["parent_header"],
            "header": candidate["header"],
            "content": candidate["content"],
            "embed_score": float(embed_score),
            "rerank_score": float(rerank_score),
            "chunk_index": candidate.get("chunk_index"),
            "source_url": candidate.get("source_url"),
        })
        logger.debug(
            f"Reranked candidate {candidate['node_id']}: embed_score={embed_score:.4f}, "
            f"rerank_score={rerank_score:.4f}")
    reranked = sorted(reranked, key=lambda x: x["score"], reverse=True)
    for rank, item in enumerate(reranked, 1):
        item["rank"] = rank
    logger.info(
        f"Reranking completed in {time.time() - start_time:.2f} seconds, "
        f"returned {len(reranked)} results")
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
    embeddings = model.encode(headers, batch_size=batch_size, show_progress_bar=False,
                              convert_to_tensor=True, device=device).cpu().numpy()

    similarities = util.cos_sim(embeddings, embeddings).cpu().numpy()
    distances = 1 - similarities

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1 - similarity_threshold,
        metric="precomputed",
        linkage="average"
    )
    labels = clustering.fit_predict(distances)

    cluster_dict = {}
    for idx, (label, text) in enumerate(zip(labels, texts)):
        if label not in cluster_dict:
            cluster_dict[label] = []
        cluster_dict[label].append({
            "original": text,
            "embedding": embeddings[idx],
            "score": float(text["score"])
        })

    deduplicated_texts: List[MergedPreprocessedText] = []
    for label, items in cluster_dict.items():
        sorted_items = sorted(items, key=lambda x: x["score"], reverse=True)
        best_doc = sorted_items[0]["original"]

        other_docs = [item["original"] for item in sorted_items[1:]]

        token_count = best_doc.get("token_count", len(
            model.tokenize([best_doc["text"]])["input_ids"][0]))
        merge_info = {
            "min_tokens": min([token_count] + [d["tokens"] for d in other_docs]) if other_docs else token_count,
            "max_tokens": max([token_count] + [d["tokens"] for d in other_docs]) if other_docs else token_count,
            "avg_tokens": float(np.mean([token_count] + [d["tokens"] for d in other_docs])) if other_docs else float(token_count)
        }

        deduplicated_texts.append({
            "text": best_doc["text"],
            "doc_index": best_doc["doc_index"],
            "node_id": best_doc["node_id"],
            "header_level": best_doc["header_level"],
            "parent_header": best_doc["parent_header"],
            "header": best_doc["header"],
            "content": best_doc["content"],
            "chunk_index": best_doc["chunk_index"],
            "token_count": token_count,
            "source_url": best_doc["source_url"],
            "merged_count": len(items),
            "merged_doc_chunk_indices": best_doc["chunk_index"],
            "merged_doc_contents": [doc["content"] for doc in other_docs],
            "merged_doc_headers": [doc["header"] for doc in other_docs],
            "merged_docs": other_docs,
            "tokens": best_doc["tokens"],
            "embed_score": best_doc["score"],
            "rerank_score": None,
            "merge_info": merge_info
        })

        logger.debug(
            f"Cluster {label}: Selected best doc {best_doc['node_id']} with embed_score={best_doc['score']:.4f}, "
            f"removed {len(other_docs)} others")

    logger.info(
        f"Reduced {len(texts)} texts to {len(deduplicated_texts)} after header-based clustering. "
        f"Time: {time.time() - start_time:.2f}s")
    return deduplicated_texts


def search_documents(
    query: str,
    headers: List[Header],
    model_name: str = "all-MiniLM-L12-v2",
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device: str = get_device(),
    top_k: int = 20,
    lambda_param: float = 0.5,
    batch_size: int = 16,
    num_threads: int = 4,
    min_header_level: int = 2,
    max_header_level: int = 6,
    parent_diversity_weight: float = 0.4,
    header_diversity_weight: float = 0.3,
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
                "score": float(lambda_param * r["rerank_score"] + (1 - lambda_param) * r["embed_score"]),
                "text": r["text"],
                "tokens": r["tokens"],
                "header_level": r["header_level"],
                "parent_header": r["parent_header"],
                "header": r["header"],
                "content": r["content"],
                "chunk_index": r.get("chunk_index"),
                "source_url": r.get("source_url"),
            } for r in rerank_results_list[:top_k]
        ]
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        for rank, item in enumerate(results, 1):
            item["rank"] = rank
        logger.info(
            f"Search completed in {time.time() - start_time:.2f} seconds, "
            f"returning {len(results)} results")
        logger.debug(
            f"Top 3 results: {[(r['header'][:30] + '...', r['embed_score'], r['rerank_score'], r['score']) for r in results[:3]]}")
        return {
            "results": results,
            "merge_results": merged_texts,
            "embed_results": embed_results,
            "rerank_results": rerank_results_list
        }
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise RuntimeError(f"Search failed: {str(e)}") from e
