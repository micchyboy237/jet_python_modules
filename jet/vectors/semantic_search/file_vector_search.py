import fnmatch
import re
import os
import numpy as np
import nbformat
from typing import List, Optional, Union, Tuple, TypedDict, Iterator, Callable
from pathlib import Path
from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding
from jet.adapters.llama_cpp.types import LLAMACPP_EMBED_KEYS, LLAMACPP_EMBED_TYPES
from tqdm import tqdm
from jet.code.markdown_utils._preprocessors import remove_markdown_links
from jet.utils.url_utils import remove_links
from jet.logger import logger

from jet.transformers.formatters import format_json
from jet.wordnet.text_chunker import chunk_texts_with_data


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
    text: str  # Changed from 'code' to 'text'


class Weights(TypedDict):
    """Typed dictionary for similarity weights."""
    name: float
    dir: float
    content: float


DEFAULT_EMBED_MODEL: LLAMACPP_EMBED_KEYS = 'embeddinggemma'

# model_context_size = LLAMACPP_MODEL_CONTEXTS[DEFAULT_EMBED_MODEL]
# # 'nomic-embed-text-v2-moe' context = 2048
# factor_1 = 0.70          # chunk_size = context * 0.70 → ~1433 tokens
# factor_2 = 0.18          # overlap   = chunk_size * 0.18 → ~258 tokens
# DEFAULT_CHUNK_SIZE = int(model_context_size * factor_1)
# DEFAULT_CHUNK_OVERLAP = int(DEFAULT_CHUNK_SIZE * factor_2)
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100

DEFAULT_WEIGHTS: Weights = {
    "dir": 0.0,
    "name": 0.25,
    "content": 0.75,
}


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return float(dot_product / (norm_a * norm_b)) if norm_a * norm_b != 0 else 0.0


def get_matched_files(
    paths: Union[str, List[str]],
    extensions: Optional[List[str]] = None,
    includes: Optional[List[str]] = None,
    excludes: Optional[List[str]] = None
) -> List[str]:
    """
    Collect file paths that match the specified extensions, includes, and excludes patterns.
    Args:
        paths: Single path or list of paths to scan
        extensions: List of file extensions to include (e.g., ['.py', '.txt'])
        includes: List of glob patterns to include
        excludes: List of glob patterns to exclude
    Returns:
        List of file paths that match the criteria
    """
    matched_paths = []
    path_list = [paths] if isinstance(paths, str) else paths

    for path in path_list:
        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist")
        if os.path.isfile(path):
            matched_paths.append(path)
        else:
            for root, _, files in os.walk(path):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    matched_paths.append(file_path)

    # Filter paths based on extensions, includes, and excludes
    filtered_paths = []
    for file_path in matched_paths:
        if extensions and not any(file_path.endswith(ext) for ext in extensions):
            continue
        if includes and not any(fnmatch.fnmatch(file_path, pattern) for pattern in includes):
            continue
        if excludes and any(fnmatch.fnmatch(file_path, pattern) for pattern in excludes):
            continue
        filtered_paths.append(file_path)

    return filtered_paths


def collect_file_contents(
    paths: Union[str, List[str]],
    extensions: Optional[List[str]] = None,
    includes: Optional[List[str]] = None,
    excludes: Optional[List[str]] = None,
    show_progress: bool = True,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    file_paths = get_matched_files(paths, extensions, includes, excludes)
    
    all_file_paths = []
    all_file_names = []
    all_file_contents = []
    all_parent_dirs = []

    # Prepare progress bar (disabled when show_progress=False)
    file_iterator = tqdm(
        file_paths,
        desc="Collecting & chunking files",
        total=len(file_paths),
        disable=not show_progress,
        unit="file",
    )

    for file_path in file_iterator:
        file_path_obj = Path(file_path)
        file_name = file_path_obj.name
        parent_dir = file_path_obj.parent.name or "root"

        suffix = file_path_obj.suffix.lower()
        if suffix == '.ipynb':
            with open(file_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            parts = []
            for cell in nb.cells:
                source = cell.get('source', '')
                if not isinstance(source, str) or not source.strip():
                    continue
                if cell.cell_type == 'markdown':
                    parts.append(source.rstrip())
                elif cell.cell_type == 'code':
                    parts.append("```python\n" + source.rstrip() + "\n```")
            if not parts:
                continue
            full_content = "\n\n".join(parts)
        
        elif suffix in {'.txt', '.py', '.md', '.mdx', '.mdc', '.rst', '.json', '.csv'}:
            with open(file_path, 'r', encoding='utf-8') as f:
                full_content = f.read()
        else:
            continue

        # Remove all links
        if suffix in {'.md', '.mdx', '.mdc', ".ipynb"}:
            full_content = remove_markdown_links(full_content, remove_text=False)
        full_content = remove_links(full_content)

        all_file_paths.append(file_path)
        all_file_names.append(file_name)
        all_parent_dirs.append(parent_dir)
        all_file_contents.append(full_content)

    return all_file_paths, all_file_names, all_file_contents, all_parent_dirs


def collect_file_chunks(
    paths: Union[str, List[str]],
    extensions: Optional[List[str]] = None,
    embed_model: "LLAMACPP_EMBED_TYPES" = DEFAULT_EMBED_MODEL,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    tokenizer: Optional[Callable[[str], int]] = None,
    includes: Optional[List[str]] = None,
    excludes: Optional[List[str]] = None,
    show_progress: bool = True,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Collect chunked contents for each file along with file paths, names, dirs, and token counts.
    Args:
        paths: Single path or list of paths to scan
        extensions: List of file extensions to include
        chunk_size: Size of content chunks
        chunk_overlap: Overlap between chunks
        tokenizer: Optional callable to count tokens in text. Defaults to regex-based word and punctuation counting.
        includes: List of glob patterns to include
        excludes: List of glob patterns to exclude
    Returns:
        Tuple of (file_paths, file_names, parent_dirs, contents_with_indices)
        where contents_with_indices = List of (file_path, content_chunk, start_idx, end_idx, num_tokens)
    """
    
    all_file_paths, all_file_names, all_texts, all_parent_dirs = collect_file_contents(paths, extensions, includes, excludes, show_progress)

    def default_tokenizer(text): 
        return len(re.findall(r'\b\w+\b|[^\w\s]', text))

    tokenizer = tokenizer or default_tokenizer

    contents_with_indices = []

    # Use chunk_texts_with_data for chunking
    chunks = chunk_texts_with_data(
        texts=all_texts,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        model=embed_model,
        ids=all_file_paths,
        buffer=0,
        show_progress=show_progress,
    )
    for chunk in chunks:
        contents_with_indices.append(
            (
                chunk["doc_id"],
                chunk['content'],
                chunk['start_idx'],
                chunk['end_idx'],
                chunk['num_tokens']
            )
        )

    return all_file_paths, all_file_names, all_parent_dirs, contents_with_indices


def compute_weighted_similarity(
    query_vector: np.ndarray,
    name_vector: np.ndarray,
    dir_vector: np.ndarray,
    content_vector: Optional[np.ndarray],
    weights: Optional[Weights] = None
) -> Tuple[float, float, float, float]:
    """
    Compute weighted similarity score and individual scores for a file based on its components.
    Args:
        query_vector: Encoded query vector
        name_vector: Encoded file name vector
        dir_vector: Encoded parent directory vector
        content_vector: Encoded content vector (if available)
        weights: Optional dictionary specifying weights for name, dir, and content similarities
    Returns:
        Tuple of (weighted_similarity, name_similarity, dir_similarity, content_similarity)
    """
    name_sim = cosine_similarity(query_vector, name_vector)
    dir_sim = cosine_similarity(query_vector, dir_vector)
    content_sim = 0.0
    if content_vector is not None:
        content_sim = cosine_similarity(query_vector, content_vector)


    # Use default weights if none provided
    active_weights = weights if weights is not None else DEFAULT_WEIGHTS

    weighted_sim = (
        active_weights["name"] * name_sim +
        active_weights["dir"] * dir_sim +
        active_weights["content"] * content_sim
    )
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
        merged_text = current_chunk["text"]
        start_idx = current_chunk["metadata"]["start_idx"]
        end_idx = current_chunk["metadata"]["end_idx"]
        max_score = current_chunk["score"]  # Track max score instead of total
        name_sim = current_chunk["metadata"]["name_similarity"]
        dir_sim = current_chunk["metadata"]["dir_similarity"]
        content_sims = [current_chunk["metadata"]["content_similarity"]]
        chunk_count = 1
        tokens = tokenizer(merged_text)

        for next_chunk in chunks[1:]:
            next_start = next_chunk["metadata"]["start_idx"]
            next_end = next_chunk["metadata"]["end_idx"]
            next_text = next_chunk["text"]
            # Check if chunks are adjacent or overlapping
            if next_start <= end_idx:
                # Extend the merged content
                new_end = max(end_idx, next_end)
                overlap = end_idx - next_start
                additional_content = next_text[overlap:] if overlap > 0 else next_text
                merged_text += additional_content
                end_idx = new_end
                # Update max score
                max_score = max(max_score, next_chunk["score"])
                content_sims.append(
                    next_chunk["metadata"]["content_similarity"])
                chunk_count += 1
                tokens = tokenizer(merged_text)
            else:
                # Finalize current merged chunk
                avg_content_sim = sum(content_sims) / chunk_count
                merged_results.append({
                    "rank": current_chunk["rank"],
                    "score": max_score,  # Use max score instead of avg_score
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
                    "text": merged_text,
                })
                # Start new merged chunk
                current_chunk = next_chunk
                merged_text = current_chunk["text"]
                start_idx = current_chunk["metadata"]["start_idx"]
                end_idx = current_chunk["metadata"]["end_idx"]
                # Reset to new chunk's score
                max_score = current_chunk["score"]
                name_sim = current_chunk["metadata"]["name_similarity"]
                dir_sim = current_chunk["metadata"]["dir_similarity"]
                content_sims = [current_chunk["metadata"]
                                ["content_similarity"]]
                chunk_count = 1
                tokens = tokenizer(merged_text)

        # Append the last merged chunk for this file
        avg_content_sim = sum(content_sims) / chunk_count
        merged_results.append({
            "rank": current_chunk["rank"],
            "score": max_score,  # Use max score instead of avg_score
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
            "text": merged_text,
        })

    # Re-sort by score to maintain ranking
    merged_results.sort(key=lambda x: x["score"], reverse=True)
    for i, result in enumerate(merged_results, 1):
        result["rank"] = i

    return merged_results


def search_files(  # type: ignore[no-untyped-def]  # temporary until full typing
    paths: Union[str, List[str]],
    query: str,
    extensions: Optional[List[str]] = None,
    top_k: Optional[int] = None,
    embed_model: "LLAMACPP_EMBED_TYPES" = DEFAULT_EMBED_MODEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    threshold: float = 0.0,
    tokenizer: Optional[Callable[[str], int]] = None,
    split_chunks: bool = False,
    includes: Optional[List[str]] = None,
    excludes: Optional[List[str]] = None,
    preprocess: Optional[Callable[[str], str]] = None,
    weights: Optional[Weights] = None,
    batch_size: int = 64,
    show_progress: bool = True,
    use_cache: bool = False,
) -> Iterator[FileSearchResult]:
    """
    Search files using vector similarity on chunked contents + file metadata.
    Yields up to top_k results iteratively that meet the threshold, or all results if top_k is None.
    Args:
        paths: Single path or list of paths to search
        query: Search query string
        extensions: List of file extensions to include
        top_k: Maximum number of results to yield, or None to yield all results
        embed_model: Embedding model name to use for vectorization (LLAMACPP_EMBED_TYPES).
        chunk_size: Size of content chunks
        chunk_overlap: Overlap between chunks
        threshold: Minimum similarity score for results
        tokenizer: Optional callable to count tokens in text
        split_chunks: If True, return individual chunks; if False, merge adjacent chunks
        includes: List of glob patterns to include
        excludes: List of glob patterns to exclude
        preprocess: Optional callback to preprocess texts before embedding
        weights: Optional dictionary specifying weights for name, dir, and content similarities
        batch_size: Batch size to use when generating embeddings
        show_progress: Display progress bar during chunking step.
    Returns:
        Iterator of FileSearchResult dictionaries (ranked by similarity)
    """
    def default_tokenizer(text): return len(
        re.findall(r'\b\w+\b|[^\w\s]', text))
    tokenizer = tokenizer or default_tokenizer

    file_paths, file_names, parent_dirs, chunk_data = collect_file_chunks(
        paths, extensions, embed_model, chunk_size, chunk_overlap, tokenizer, includes, excludes, show_progress=show_progress)
    logger.debug(f"Parent dirs:\n\n{format_json(parent_dirs)}")
    logger.debug(f"File names:\n\n{format_json(file_names)}")
    logger.debug(f"File paths:\n\n{format_json(file_paths)}")
    if not chunk_data:
        return
    unique_files = list(dict.fromkeys(file_paths))
    name_texts = [Path(p).name for p in unique_files]
    dir_texts = [Path(p).parent.name or "root" for p in unique_files]
    chunk_texts = [chunk for _, chunk, _, _, _ in chunk_data]

    # ── 1. Embed query (instant) ────────────────────────
    embedder = LlamacppEmbedding(model=embed_model, use_cache=use_cache, verbose=True, cache_ttl=86400)
    query_processed = preprocess(query) if preprocess else query
    query_vector: np.ndarray = embedder(
        query_processed,
        return_format="numpy",
        batch_size=1,
        show_progress=False,
        use_cache=use_cache,
    )[0]

    # ── 2. Embed names + dirs (small → fast) ────────────
    processed_name_texts = [preprocess(name) if preprocess else name for name in name_texts]
    processed_dir_texts = [preprocess(dir_name) if preprocess else dir_name for dir_name in dir_texts]

    name_dir_texts = processed_name_texts + processed_dir_texts
    if name_dir_texts:
        name_dir_vectors: np.ndarray = embedder(
            name_dir_texts,
            return_format="numpy",
            batch_size=min(128, len(name_dir_texts)),
            show_progress=True,
            use_cache=use_cache,
        )
        name_vectors = name_dir_vectors[:len(processed_name_texts)]
        dir_vectors = name_dir_vectors[len(processed_name_texts):]
    else:
        name_vectors = np.array([])
        dir_vectors = np.array([])

    # ── 3. Stream chunk embeddings ──────────────────────
    processed_chunk_texts = [preprocess(c) if preprocess else c for c in chunk_texts]

    logger.info(f"Streaming embeddings for {len(processed_chunk_texts)} chunks...")

    chunk_embeddings_stream = embedder.get_embeddings_stream(
        processed_chunk_texts,
        return_format="numpy",
        batch_size=batch_size,
        show_progress=True,
        use_cache=use_cache,
    )

    results: List[FileSearchResult] = []
    chunk_counts = {}

    # We'll collect results in a list and yield progressively
    # (could use heapq for top-k early cutoff if top_k is small)

    yielded = 0
    for batch_idx, batch_vectors in enumerate(chunk_embeddings_stream):
        logger.debug(f"Received chunk embedding batch {batch_idx+1} — {len(batch_vectors)} vectors")

        batch_start = batch_idx * batch_size
        batch_end   = batch_start + len(batch_vectors)

        for local_i, content_vector in enumerate(batch_vectors):
            global_i = batch_start + local_i
            if global_i >= len(chunk_data):
                break  # in case the last batch is smaller than batch_size

            file_path, chunk, start_idx, end_idx, num_tokens = chunk_data[global_i]
            file_index = unique_files.index(file_path)

            weighted_sim, name_sim, dir_sim, content_sim = compute_weighted_similarity(
                query_vector,
                name_vectors[file_index],
                dir_vectors[file_index],
                content_vector,
                weights
            )

            if weighted_sim >= threshold:
                chunk_counts[file_path] = chunk_counts.get(file_path, -1) + 1

                result: FileSearchResult = {
                    "rank": 0,  # temporary — re-ranked later if needed
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
                    "text": chunk,
                }
                results.append(result)

                # Yield immediately (progressive feedback)
                # Note: rank is not final yet — can be post-processed if desired
                yielded += 1
                if top_k is None or yielded <= top_k:
                    yield result
                if top_k is not None and yielded >= top_k:
                    return  # stop yielding if reached top_k

    # Final sorting & rank assignment if you want complete list at the end
    results.sort(key=lambda x: x["score"], reverse=True)
    for i, r in enumerate(results, 1):
        r["rank"] = i

    # If split_chunks=False → could run merge_results(...) here
    # but since we want streaming → merging usually happens later / optionally

    if not split_chunks:
        merged_results = merge_results(results, tokenizer)
        for i, result in enumerate(merged_results if top_k is None else merged_results[:top_k], 1):
            result["rank"] = i
            yield result
