from typing import List, Optional, Union, Tuple, TypedDict, Iterator
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


class FileSearchResult(TypedDict):
    """Typed dictionary for search result structure."""
    rank: int
    score: float
    code: str
    metadata: FileSearchMetadata


def get_file_vectors(file_path: str, embed_model: EmbedModelType = DEFAULT_EMBED_MODEL) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Generate vector representations for file name, parent directory, and content using SentenceTransformer."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist")
    model = SentenceTransformerRegistry.load_model(embed_model)
    file_path_obj = Path(file_path)
    file_name = file_path_obj.name.lower()
    parent_dir = file_path_obj.parent.name.lower(
    ) if file_path_obj.parent.name else "root"
    file_name_vector = model.encode(file_name, convert_to_numpy=True)
    parent_dir_vector = model.encode(parent_dir, convert_to_numpy=True)
    content_vector = None
    try:
        if file_path_obj.suffix.lower() in {'.txt', '.py', '.md', '.json', '.csv'}:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(MAX_CONTENT_SIZE).lower()
                if content.strip():
                    content_vector = model.encode(
                        content, convert_to_numpy=True)
    except (UnicodeDecodeError, IOError):
        pass
    return file_name_vector, parent_dir_vector, content_vector


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
    chunk_overlap: int = 100
) -> Tuple[List[str], List[str], List[str], List[Tuple[str, str, int, int]]]:
    """
    Collect chunked contents for each file along with file paths, names, and dirs.
    Returns:
        Tuple of (file_paths, file_names, parent_dirs, contents_with_indices)
        where contents_with_indices = List of (file_path, content_chunk, start_idx, end_idx)
    """
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
                                contents_with_indices.append(
                                    (file_path, chunk, start, end))
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


def merge_results(results: List[FileSearchResult]) -> List[FileSearchResult]:
    """
    Merge adjacent chunks from the same file into a single result, preserving order and metadata.
    Args:
        results: List of FileSearchResult dictionaries, potentially containing adjacent chunks.
    Returns:
        List of FileSearchResult dictionaries with adjacent chunks merged.
    """
    if not results:
        return []

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

        current_chunk = chunks[0]
        merged_code = current_chunk["code"]
        start_idx = current_chunk["metadata"]["start_idx"]
        end_idx = current_chunk["metadata"]["end_idx"]
        total_score = current_chunk["score"]
        name_sim = current_chunk["metadata"]["name_similarity"]
        dir_sim = current_chunk["metadata"]["dir_similarity"]
        content_sims = [current_chunk["metadata"]["content_similarity"]]
        chunk_count = 1

        for next_chunk in chunks[1:]:
            # Check if chunks are adjacent or overlapping
            if next_chunk["metadata"]["start_idx"] <= end_idx:
                # Extend the merged content
                new_end = max(end_idx, next_chunk["metadata"]["end_idx"])
                additional_content = next_chunk["code"][end_idx -
                                                        next_chunk["metadata"]["start_idx"]:]
                merged_code += additional_content
                end_idx = new_end
                total_score += next_chunk["score"]
                content_sims.append(
                    next_chunk["metadata"]["content_similarity"])
                chunk_count += 1
            else:
                # Finalize current merged chunk
                avg_score = total_score / chunk_count
                avg_content_sim = sum(content_sims) / chunk_count
                merged_results.append({
                    "rank": current_chunk["rank"],
                    "score": avg_score,
                    "code": merged_code,
                    "metadata": {
                        "file_path": file_path,
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "chunk_idx": 0,  # Merged chunk gets index 0
                        "name_similarity": name_sim,
                        "dir_similarity": dir_sim,
                        "content_similarity": avg_content_sim
                    }
                })
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

        # Append the last merged chunk for this file
        avg_score = total_score / chunk_count
        avg_content_sim = sum(content_sims) / chunk_count
        merged_results.append({
            "rank": current_chunk["rank"],
            "score": avg_score,
            "code": merged_code,
            "metadata": {
                "file_path": file_path,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "chunk_idx": 0,
                "name_similarity": name_sim,
                "dir_similarity": dir_sim,
                "content_similarity": avg_content_sim
            }
        })

    # Re-sort by score to maintain ranking
    merged_results.sort(key=lambda x: x["score"], reverse=True)
    for i, result in enumerate(merged_results, 1):
        result["rank"] = i

    return merged_results


def search_files(
    paths: Union[str, List[str]],
    query: str,
    extensions: Optional[List[str]] = None,
    top_k: int = 5,
    embed_model: EmbedModelType = DEFAULT_EMBED_MODEL,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    threshold: float = 0.0
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
    Returns:
        Iterator of FileSearchResult dictionaries (ranked by similarity)
    """
    file_paths, file_names, parent_dirs, chunk_data = collect_file_chunks(
        paths, extensions, chunk_size, chunk_overlap)

    if not chunk_data:
        return

    unique_files = list(dict.fromkeys(file_paths))
    name_texts = [Path(p).name.lower() for p in unique_files]
    dir_texts = [Path(p).parent.name.lower() or "root" for p in unique_files]
    chunk_texts = [chunk for _, chunk, _, _ in chunk_data]

    # Combine query with all texts for one embedding call
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

    # Split embeddings back into respective groups
    query_vector = all_vectors[0]
    num_names = len(name_texts)
    num_dirs = len(dir_texts)
    name_vectors = all_vectors[1:num_names + 1]
    dir_vectors = all_vectors[num_names + 1:num_names + 1 + num_dirs]
    content_vectors = all_vectors[num_names + 1 + num_dirs:]

    results: List[FileSearchResult] = []
    chunk_counts = {}  # Track chunk index per file
    for i, (file_path, chunk, start_idx, end_idx) in enumerate(chunk_data):
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
                "code": chunk,
                "metadata": {
                    "file_path": file_path,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "chunk_idx": chunk_counts[file_path],
                    "name_similarity": float(name_sim),
                    "dir_similarity": float(dir_sim),
                    "content_similarity": float(content_sim)
                }
            }
            results.append(result)

    results.sort(key=lambda x: x["score"], reverse=True)
    for i, result in enumerate(results[:top_k], 1):
        result["rank"] = i
        yield result
