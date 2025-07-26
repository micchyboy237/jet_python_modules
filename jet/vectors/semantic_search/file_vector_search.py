import re
from typing import List, Optional, Union, Tuple, TypedDict, Iterator, Callable
import os
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from jet.logger import logger
from jet.models.embeddings.base import generate_embeddings
from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry
from jet.models.model_types import EmbedModelType
import logging

DEFAULT_EMBED_MODEL: EmbedModelType = 'all-MiniLM-L6-v2'
MAX_CONTENT_SIZE = 1000


class FileSearchMetadata(TypedDict):
    """Typed dictionary for search result metadata."""
    file_path: str
    start_idx: int
    end_idx: int
    chunk_idx: int
    name_similarity: float
    dir_similarity: float
    content_similarity: float
    num_tokens: int


class FileSearchResult(TypedDict):
    """Typed dictionary for search result structure."""
    rank: int
    score: float
    metadata: FileSearchMetadata
    code: str


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_product / (norm_a * norm_b) if norm_a * norm_b != 0 else 0.0


def collect_file_chunks(
    paths: Union[str, List[str]],
    extensions: List[str] = None,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    tokenizer: Optional[Callable[[str], int]] = None
) -> Tuple[List[str], List[str], List[str], List[Tuple[str, str, int, int, int]]]:
    """
    Collect chunked contents for each file along with file paths, names, dirs, and token counts.
    Args:
        paths: Single path or list of paths to scan
        extensions: List of file extensions to include
        chunk_size: Size of content chunks
        chunk_overlap: Overlap between chunks
        tokenizer: Optional callable to count tokens in text. Defaults to regex-based word and punctuation counting.
    Returns:
        Tuple of (file_paths, file_names, parent_dirs, contents_with_indices)
        where contents_with_indices = List of (file_path, content_chunk, start_idx, end_idx, num_tokens)
    """
    def default_tokenizer(text): return len(
        re.findall(r'\b\w+\b|[^\w\s]', text))
    tokenizer = tokenizer or default_tokenizer

    file_paths, file_names, parent_dirs = [], [], []
    contents_with_indices = []
    path_list = [paths] if isinstance(paths, str) else paths
    for path in path_list:
        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist")
        paths_to_scan = [path] if os.path.isfile(path) else [
            os.path.join(root, f)
            for root, _, files in os.walk(path)
            for f in files
        ]
        for file_path in paths_to_scan:
            if extensions and not any(file_path.lower().endswith(ext) for ext in extensions):
                continue
            file_path_obj = Path(file_path)
            file_paths.append(file_path)
            file_names.append(file_path_obj.name.lower())
            parent_dirs.append(file_path_obj.parent.name.lower() or "root")
            try:
                if file_path_obj.suffix.lower() in {'.txt', '.py', '.md', '.json', '.csv'}:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        full_content = f.read().lower()
                        for start in range(0, len(full_content), chunk_size - chunk_overlap):
                            chunk = full_content[start:start + chunk_size]
                            if chunk.strip():
                                end = start + len(chunk)
                                num_tokens = tokenizer(chunk)  # Count tokens
                                contents_with_indices.append(
                                    (file_path, chunk, start, end, num_tokens))
            except (UnicodeDecodeError, IOError):
                continue
    return file_paths, file_names, parent_dirs, contents_with_indices


def compute_weighted_similarity(
    query_vector: np.ndarray,
    name_vector: np.ndarray,
    dir_vector: np.ndarray,
    content_vector: Optional[np.ndarray]
) -> Tuple[float, float, float, float]:
    """
    Compute weighted similarity score and individual scores for a file based on its components.
    Args:
        query_vector: Encoded query vector
        name_vector: Encoded file name vector
        dir_vector: Encoded parent directory vector
        content_vector: Encoded content vector (if available)
    Returns:
        Tuple of (weighted_similarity, name_similarity, dir_similarity, content_similarity)
    """
    name_sim = cosine_similarity(query_vector, name_vector)
    dir_sim = cosine_similarity(query_vector, dir_vector)
    content_sim = 0.0
    if content_vector is not None:
        content_sim = cosine_similarity(query_vector, content_vector)
    weighted_sim = 0.4 * name_sim + 0.2 * dir_sim + 0.4 * content_sim
    return weighted_sim, name_sim, dir_sim, content_sim


def merge_results(
    results: List[FileSearchResult],
    tokenizer: Optional[Callable[[str], int]] = None
) -> List[FileSearchResult]:
    """
    Merge adjacent chunks from the same file into a single result, preserving order and metadata.
    Args:
        results: List of FileSearchResult dictionaries, potentially containing adjacent chunks.
        tokenizer: Optional callable to count tokens in text. Defaults to regex-based word and punctuation counting.
    Returns:
        List of FileSearchResult dictionaries with adjacent chunks merged.
    """
    if not results:
        return []

    # Default tokenizer if none provided
    def default_tokenizer(text): return len(
        re.findall(r'\b\w+\b|[^\w\s]', text))
    tokenizer = tokenizer or default_tokenizer

    logger.debug(f"Starting merge_results with {len(results)} results")
    # Group results by file_path
    grouped: dict[str, List[FileSearchResult]] = {}
    for result in results:
        file_path = result["metadata"]["file_path"]
        if file_path not in grouped:
            grouped[file_path] = []
        grouped[file_path].append(result)

    merged_results: List[FileSearchResult] = []
    for file_path, chunks in grouped.items():
        # Sort chunks by start_idx to ensure correct order
        chunks.sort(key=lambda x: x["metadata"]["start_idx"])
        logger.debug(f"Processing file: {file_path}, {len(chunks)} chunks")

        current_chunk = chunks[0]
        merged_code = current_chunk["code"]
        start_idx = current_chunk["metadata"]["start_idx"]
        end_idx = current_chunk["metadata"]["end_idx"]
        total_score = current_chunk["score"]
        name_sim = current_chunk["metadata"]["name_similarity"]
        dir_sim = current_chunk["metadata"]["dir_similarity"]
        content_sims = [current_chunk["metadata"]["content_similarity"]]
        chunk_count = 1
        tokens = tokenizer(merged_code)
        logger.debug(
            f"Initial chunk: start_idx={start_idx}, end_idx={end_idx}, code='{merged_code}', tokens={tokens}")

        for next_chunk in chunks[1:]:
            next_start = next_chunk["metadata"]["start_idx"]
            next_end = next_chunk["metadata"]["end_idx"]
            next_code = next_chunk["code"]
            logger.debug(
                f"Next chunk: start_idx={next_start}, end_idx={next_end}, code='{next_code}'")
            # Check if chunks are adjacent or overlapping
            if next_start <= end_idx:
                # Extend the merged content
                new_end = max(end_idx, next_end)
                overlap = end_idx - next_start
                additional_content = next_code[overlap:] if overlap > 0 else next_code
                logger.debug(
                    f"Merging: overlap={overlap}, additional_content='{additional_content}'")
                merged_code += additional_content
                end_idx = new_end
                total_score += next_chunk["score"]
                content_sims.append(
                    next_chunk["metadata"]["content_similarity"])
                chunk_count += 1
                tokens = tokenizer(merged_code)
                logger.debug(
                    f"After merge: merged_code='{merged_code}', tokens={tokens}")
            else:
                # Finalize current merged chunk
                avg_score = total_score / chunk_count
                avg_content_sim = sum(content_sims) / chunk_count
                merged_results.append({
                    "rank": current_chunk["rank"],
                    "score": avg_score,
                    "metadata": {
                        "file_path": file_path,
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "chunk_idx": 0,
                        "name_similarity": name_sim,
                        "dir_similarity": dir_sim,
                        "content_similarity": avg_content_sim,
                        "num_tokens": tokens
                    },
                    "code": merged_code,
                })
                logger.debug(
                    f"Finalized chunk: start_idx={start_idx}, end_idx={end_idx}, tokens={tokens}")
                # Start new merged chunk
                current_chunk = next_chunk
                merged_code = current_chunk["code"]
                start_idx = current_chunk["metadata"]["start_idx"]
                end_idx = current_chunk["metadata"]["end_idx"]
                total_score = current_chunk["score"]
                name_sim = current_chunk["metadata"]["name_similarity"]
                dir_sim = current_chunk["metadata"]["dir_similarity"]
                content_sims = [current_chunk["metadata"]
                                ["content_similarity"]]
                chunk_count = 1
                tokens = tokenizer(merged_code)
                logger.debug(
                    f"New chunk started: start_idx={start_idx}, end_idx={end_idx}, code='{merged_code}', tokens={tokens}")

        # Append the last merged chunk for this file
        avg_score = total_score / chunk_count
        avg_content_sim = sum(content_sims) / chunk_count
        merged_results.append({
            "rank": current_chunk["rank"],
            "score": avg_score,
            "metadata": {
                "file_path": file_path,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "chunk_idx": 0,
                "name_similarity": name_sim,
                "dir_similarity": dir_sim,
                "content_similarity": avg_content_sim,
                "num_tokens": tokens
            },
            "code": merged_code,
        })
        logger.debug(
            f"Appended final chunk: start_idx={start_idx}, end_idx={end_idx}, tokens={tokens}")

    # Re-sort by score to maintain ranking
    merged_results.sort(key=lambda x: x["score"], reverse=True)
    for i, result in enumerate(merged_results, 1):
        result["rank"] = i
    logger.debug(f"Merged results: {len(merged_results)} results")

    return merged_results


def search_files(
    paths: Union[str, List[str]],
    query: str,
    extensions: Optional[List[str]] = None,
    top_k: int = 5,
    embed_model: EmbedModelType = DEFAULT_EMBED_MODEL,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    threshold: float = 0.0,
    tokenizer: Optional[Callable[[str], int]] = None,
    split_chunks: bool = False
) -> Iterator[FileSearchResult]:
    """
    Search files using vector similarity on chunked contents + file metadata.
    Yields up to top_k results iteratively that meet the threshold.
    Args:
        paths: Single path or list of paths to search
        query: Search query string
        extensions: List of file extensions to include
        top_k: Maximum number of results to yield
        embed_model: Embedding model to use
        chunk_size: Size of content chunks
        chunk_overlap: Overlap between chunks
        threshold: Minimum similarity score for results
        tokenizer: Optional callable to count tokens in text. Defaults to regex-based word and punctuation counting.
        split_chunks: If True, return individual chunks; if False, merge adjacent chunks from the same file.
    Returns:
        Iterator of FileSearchResult dictionaries (ranked by similarity)
    """
    # Default tokenizer if none provided
    def default_tokenizer(text): return len(
        re.findall(r'\b\w+\b|[^\w\s]', text))
    tokenizer = tokenizer or default_tokenizer

    file_paths, file_names, parent_dirs, chunk_data = collect_file_chunks(
        paths, extensions, chunk_size, chunk_overlap, tokenizer)
    if not chunk_data:
        return
    unique_files = list(dict.fromkeys(file_paths))
    name_texts = [Path(p).name.lower() for p in unique_files]
    dir_texts = [Path(p).parent.name.lower() or "root" for p in unique_files]
    chunk_texts = [chunk for _, chunk, _, _, _ in chunk_data]
    all_texts = [query] + name_texts + dir_texts + chunk_texts
    logger.info(
        f"Generating embeddings for {len(all_texts)} texts:\n"
        f"  1 query\n"
        f"  {len(name_texts)} names\n"
        f"  {len(dir_texts)} dirs\n"
        f"  {len(chunk_texts)} chunks"
    )
    all_vectors = generate_embeddings(
        all_texts,
        embed_model,
        return_format="numpy",
        batch_size=32,
        show_progress=True
    )
    query_vector = all_vectors[0]
    num_names = len(name_texts)
    num_dirs = len(dir_texts)
    name_vectors = all_vectors[1:num_names + 1]
    dir_vectors = all_vectors[num_names + 1:num_names + 1 + num_dirs]
    content_vectors = all_vectors[num_names + 1 + num_dirs:]
    results: List[FileSearchResult] = []
    chunk_counts = {}
    for i, (file_path, chunk, start_idx, end_idx, num_tokens) in enumerate(chunk_data):
        file_index = unique_files.index(file_path)
        content_vector = content_vectors[i]
        weighted_sim, name_sim, dir_sim, content_sim = compute_weighted_similarity(
            query_vector, name_vectors[file_index], dir_vectors[file_index], content_vector
        )
        if weighted_sim >= threshold:
            chunk_counts[file_path] = chunk_counts.get(file_path, -1) + 1
            result = {
                "rank": 0,
                "score": float(weighted_sim),
                "metadata": {
                    "file_path": file_path,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "chunk_idx": chunk_counts[file_path],
                    "name_similarity": float(name_sim),
                    "dir_similarity": float(dir_sim),
                    "content_similarity": float(content_sim),
                    "num_tokens": num_tokens
                },
                "code": chunk,
            }
            results.append(result)
    results.sort(key=lambda x: x["score"], reverse=True)
    if not split_chunks:
        results = merge_results(results, tokenizer)
    for i, result in enumerate(results[:top_k], 1):
        result["rank"] = i
        yield result
