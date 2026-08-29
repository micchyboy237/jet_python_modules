import fnmatch
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TypedDict

import nbformat
import numpy as np
from jet.adapters.llama_cpp.chunking_utils import chunk_texts_with_data
from jet.adapters.llama_cpp.config import (
    EMBED_DOC_PREFIX,
    EMBED_MODEL_LG,
    EMBED_QUERY_PREFIX,
)
from jet.adapters.llama_cpp.embed_utils import embed
from jet.adapters.llama_cpp.model_utils import get_model_ctx_embd_size
from jet.adapters.llama_cpp.token_utils import count_tokens
from jet.adapters.llama_cpp.types import LLAMACPP_EMBED_KEYS, LLAMACPP_EMBED_TYPES
from jet.code.markdown_utils._preprocessors import remove_markdown_links
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.utils.url_utils import remove_links
from tqdm import tqdm


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
    text: str


class Weights(TypedDict):
    """Typed dictionary for similarity weights."""

    name: float
    dir: float
    content: float


DEFAULT_EMBED_MODEL: LLAMACPP_EMBED_KEYS = EMBED_MODEL_LG
DEFAULT_QUERY_PREFIX = EMBED_QUERY_PREFIX
DEFAULT_DOC_PREFIX = EMBED_DOC_PREFIX
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_WEIGHTS: Weights = {
    "dir": 0.10,
    "name": 0.20,
    "content": 0.70,
}


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    if vec1.size == 0 or vec2.size == 0:
        return 0.0
    if vec1.shape[0] != vec2.shape[0]:
        logger.warning(f"Vector dimension mismatch: {vec1.shape} vs {vec2.shape}")
        return 0.0
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return float(dot_product / (norm_a * norm_b)) if norm_a * norm_b != 0 else 0.0


def get_max_context_size(model_name: str) -> int:
    """
    Get the maximum context size for the embedding model.

    Uses existing get_model_ctx_embd_size utility.

    Args:
        model_name: Name of the embedding model

    Returns:
        Maximum context size in tokens (n_ctx from model metadata)
    """
    try:
        model_info = get_model_ctx_embd_size(model_name)
        return model_info["ctx"]
    except Exception as e:
        logger.warning(
            f"Failed to get context size for {model_name}: {e}. Using default 512."
        )
        return 512


def get_matched_files(
    paths: str | list[str],
    extensions: list[str] | None = None,
    includes: list[str] | None = None,
    excludes: list[str] | None = None,
) -> list[str]:
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

    filtered_paths = []
    for file_path in matched_paths:
        if extensions and not any(file_path.endswith(ext) for ext in extensions):
            continue
        if includes and not any(
            fnmatch.fnmatch(file_path, pattern) for pattern in includes
        ):
            continue
        if excludes and any(
            fnmatch.fnmatch(file_path, pattern) for pattern in excludes
        ):
            continue
        filtered_paths.append(file_path)

    return filtered_paths


def collect_file_contents(
    paths: str | list[str],
    extensions: list[str] | None = None,
    includes: list[str] | None = None,
    excludes: list[str] | None = None,
    show_progress: bool = True,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    Collect raw contents from matched files.

    Returns:
        Tuple of (file_paths, file_names, file_contents, parent_dirs)
    """
    file_paths = get_matched_files(paths, extensions, includes, excludes)

    all_file_paths = []
    all_file_names = []
    all_file_contents = []
    all_parent_dirs = []

    file_iterator = tqdm(
        file_paths,
        desc="Collecting & chunking files",
        total=len(file_paths),
        disable=not show_progress,
        unit="file",
    )

    skipped_files = 0
    encoding_errors = 0

    for file_path in file_iterator:
        file_path_obj = Path(file_path)
        file_name = file_path_obj.name
        parent_dir = file_path_obj.parent.name or "root"
        suffix = file_path_obj.suffix.lower()

        try:
            if suffix == ".ipynb":
                with open(file_path, encoding="utf-8") as f:
                    nb = nbformat.read(f, as_version=4)

                parts = []
                for cell in nb.cells:
                    source = cell.get("source", "")
                    if not isinstance(source, str) or not source.strip():
                        continue
                    if cell.cell_type == "markdown":
                        parts.append(source.rstrip())
                    elif cell.cell_type == "code":
                        parts.append("```python\n" + source.rstrip() + "\n```")

                if not parts:
                    skipped_files += 1
                    continue

                full_content = "\n\n".join(parts)

            elif suffix in {
                ".txt",
                ".py",
                ".md",
                ".mdx",
                ".mdc",
                ".rst",
                ".json",
                ".csv",
            }:
                try:
                    with open(file_path, encoding="utf-8") as f:
                        full_content = f.read()
                except UnicodeDecodeError:
                    logger.warning(
                        f"UTF-8 decode failed for {file_path}, trying latin-1 encoding"
                    )
                    try:
                        with open(file_path, encoding="latin-1") as f:
                            full_content = f.read()
                    except UnicodeDecodeError:
                        logger.warning(
                            f"latin-1 decode also failed for {file_path}, using 'replace' error handler"
                        )
                        with open(file_path, encoding="utf-8", errors="replace") as f:
                            full_content = f.read()
                        encoding_errors += 1

            elif suffix == ".pdf":
                skipped_files += 1
                logger.debug(f"Skipping PDF file: {file_path}")
                continue
            else:
                skipped_files += 1
                logger.debug(f"Skipping unsupported file type: {file_path}")
                continue

            if suffix in {".md", ".mdx", ".mdc", ".ipynb"}:
                full_content = remove_markdown_links(full_content, remove_text=False)
            full_content = remove_links(full_content)

            all_file_paths.append(file_path)
            all_file_names.append(file_name)
            all_parent_dirs.append(parent_dir)
            all_file_contents.append(full_content)

        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            skipped_files += 1
            continue

    if skipped_files > 0:
        logger.info(f"Skipped {skipped_files} files (unsupported type or read error)")
    if encoding_errors > 0:
        logger.warning(
            f"{encoding_errors} files had encoding issues and were read with character replacement"
        )

    return all_file_paths, all_file_names, all_file_contents, all_parent_dirs


def collect_file_chunks(
    paths: str | list[str],
    extensions: list[str] | None = None,
    embed_model: "LLAMACPP_EMBED_TYPES" = DEFAULT_EMBED_MODEL,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    tokenizer: Callable[[str], int] | None = None,
    includes: list[str] | None = None,
    excludes: list[str] | None = None,
    show_progress: bool = True,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    Collect chunked contents for each file along with file paths, names, dirs, and token counts.

    Args:
        paths: Single path or list of paths to scan
        extensions: List of file extensions to include
        embed_model: Embedding model name (used for chunk sizing)
        chunk_size: Size of content chunks
        chunk_overlap: Overlap between chunks
        tokenizer: Optional callable to count tokens in text. Uses llama_cpp count_tokens by default.
        includes: List of glob patterns to include
        excludes: List of glob patterns to exclude
        show_progress: Display progress bar

    Returns:
        Tuple of (file_paths, file_names, parent_dirs, contents_with_indices)
        where contents_with_indices = List of (file_path, content_chunk, start_idx, end_idx, num_tokens)
    """
    all_file_paths, all_file_names, all_texts, all_parent_dirs = collect_file_contents(
        paths, extensions, includes, excludes, show_progress
    )

    def default_tokenizer(text):
        """Use llama_cpp count_tokens for accurate token counting."""
        return count_tokens(text)

    tokenizer = tokenizer or default_tokenizer

    max_context_size = get_max_context_size(embed_model)
    logger.info(f"Model {embed_model} max context size: {max_context_size} tokens")

    if chunk_size > max_context_size:
        logger.warning(
            f"Chunk size {chunk_size} exceeds model context {max_context_size}. "
            f"Adjusting to {max_context_size}"
        )
        chunk_size = max_context_size - chunk_overlap

    contents_with_indices = []

    chunks = chunk_texts_with_data(
        texts=all_texts,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        model=embed_model,
        ids=all_file_paths,
        buffer=0,
        show_progress=show_progress,
    )

    skipped_chunks = 0
    for chunk in chunks:
        num_tokens = tokenizer(chunk["content"])
        if num_tokens > max_context_size:
            logger.warning(
                f"Skipping chunk with {num_tokens} tokens (exceeds max context {max_context_size})"
            )
            skipped_chunks += 1
            continue
        contents_with_indices.append(
            (
                chunk["doc_id"],
                chunk["content"],
                chunk["start_idx"],
                chunk["end_idx"],
                num_tokens,
            )
        )

    if skipped_chunks > 0:
        logger.warning(
            f"Skipped {skipped_chunks} chunks that exceeded context size limit"
        )

    return all_file_paths, all_file_names, all_parent_dirs, contents_with_indices


def compute_weighted_similarity(
    query_vector: np.ndarray,
    name_vector: np.ndarray,
    dir_vector: np.ndarray,
    content_vector: np.ndarray | None,
    weights: Weights | None = None,
) -> tuple[float, float, float, float]:
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

    active_weights = weights if weights is not None else DEFAULT_WEIGHTS

    weighted_sim = (
        active_weights["name"] * name_sim
        + active_weights["dir"] * dir_sim
        + active_weights["content"] * content_sim
    )

    return weighted_sim, name_sim, dir_sim, content_sim


def merge_results(
    results: list[FileSearchResult], tokenizer: Callable[[str], int] | None = None
) -> list[FileSearchResult]:
    """
    Merge adjacent chunks from the same file into a single result, preserving order and metadata.

    Args:
        results: List of FileSearchResult dictionaries, potentially containing adjacent chunks.
        tokenizer: Optional callable to count tokens in text. Uses llama_cpp count_tokens by default.

    Returns:
        List of FileSearchResult dictionaries with adjacent chunks merged.
    """
    if not results:
        return []

    def default_tokenizer(text):
        """Use llama_cpp count_tokens for accurate token counting."""
        return count_tokens(text)

    tokenizer = tokenizer or default_tokenizer

    grouped: dict[str, list[FileSearchResult]] = {}
    for result in results:
        file_path = result["metadata"]["file_path"]
        if file_path not in grouped:
            grouped[file_path] = []
        grouped[file_path].append(result)

    merged_results: list[FileSearchResult] = []

    for file_path, chunks in grouped.items():
        chunks.sort(key=lambda x: x["metadata"]["start_idx"])

        current_chunk = chunks[0]
        merged_text = current_chunk["text"]
        start_idx = current_chunk["metadata"]["start_idx"]
        end_idx = current_chunk["metadata"]["end_idx"]
        max_score = current_chunk["score"]
        name_sim = current_chunk["metadata"]["name_similarity"]
        dir_sim = current_chunk["metadata"]["dir_similarity"]
        content_sims = [current_chunk["metadata"]["content_similarity"]]
        chunk_count = 1
        tokens = tokenizer(merged_text)

        for next_chunk in chunks[1:]:
            next_start = next_chunk["metadata"]["start_idx"]
            next_end = next_chunk["metadata"]["end_idx"]
            next_text = next_chunk["text"]

            if next_start <= end_idx:
                new_end = max(end_idx, next_end)
                overlap = end_idx - next_start
                additional_content = next_text[overlap:] if overlap > 0 else next_text
                merged_text += additional_content
                end_idx = new_end
                max_score = max(max_score, next_chunk["score"])
                content_sims.append(next_chunk["metadata"]["content_similarity"])
                chunk_count += 1
                tokens = tokenizer(merged_text)
            else:
                avg_content_sim = sum(content_sims) / chunk_count
                merged_results.append(
                    {
                        "rank": current_chunk["rank"],
                        "score": max_score,
                        "metadata": {
                            "file_path": file_path,
                            "start_idx": start_idx,
                            "end_idx": end_idx,
                            "chunk_idx": 0,
                            "name_similarity": name_sim,
                            "dir_similarity": dir_sim,
                            "content_similarity": avg_content_sim,
                            "num_tokens": tokens,
                        },
                        "text": merged_text,
                    }
                )

                current_chunk = next_chunk
                merged_text = current_chunk["text"]
                start_idx = current_chunk["metadata"]["start_idx"]
                end_idx = current_chunk["metadata"]["end_idx"]
                max_score = current_chunk["score"]
                name_sim = current_chunk["metadata"]["name_similarity"]
                dir_sim = current_chunk["metadata"]["dir_similarity"]
                content_sims = [current_chunk["metadata"]["content_similarity"]]
                chunk_count = 1
                tokens = tokenizer(merged_text)

        avg_content_sim = sum(content_sims) / chunk_count
        merged_results.append(
            {
                "rank": current_chunk["rank"],
                "score": max_score,
                "metadata": {
                    "file_path": file_path,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "chunk_idx": 0,
                    "name_similarity": name_sim,
                    "dir_similarity": dir_sim,
                    "content_similarity": avg_content_sim,
                    "num_tokens": tokens,
                },
                "text": merged_text,
            }
        )

    merged_results.sort(key=lambda x: x["score"], reverse=True)
    for i, result in enumerate(merged_results, 1):
        result["rank"] = i

    return merged_results


def search_files(
    paths: str | list[str],
    query: str,
    extensions: list[str] | None = None,
    top_k: int | None = None,
    embed_model: "LLAMACPP_EMBED_TYPES" = DEFAULT_EMBED_MODEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    threshold: float = 0.0,
    tokenizer: Callable[[str], int] | None = None,
    split_chunks: bool = False,
    includes: list[str] | None = None,
    excludes: list[str] | None = None,
    preprocess: Callable[[str], str] | None = None,
    weights: Weights | None = None,
    batch_size: int = 64,
    max_workers: int = 2,
    show_progress: bool = True,
    use_cache: bool = False,
    query_prefix: str = DEFAULT_QUERY_PREFIX,
    doc_prefix: str = DEFAULT_DOC_PREFIX,
) -> Iterator[FileSearchResult]:
    """
    Search files using vector similarity on chunked contents + file metadata.

    Yields up to top_k results iteratively that meet the threshold, or all
    results if top_k is None. Uses a remote llama.cpp server for embedding
    generation and token counting.

    Query text is embedded with EMBED_QUERY_PREFIX and file name/dir/content
    text is embedded with EMBED_DOC_PREFIX (both from
    jet.adapters.llama_cpp.config), matching the asymmetric prefix convention
    expected by nomic-embed-style models.

    Remote Tuning Guide (Mac M1 → Windows PC):
        batch_size:
            - 64 (default): Balanced for WiFi and LAN
            - 128+: Use on wired gigabit ethernet only
            - 32: Use if experiencing timeouts on slow WiFi
        max_workers:
            - 2 (default): Safe for WiFi, prevents TCP saturation
            - 4: Use on wired gigabit with 8GB+ VRAM GPU
            - 1: Use if server is shared or on congested networks

    Args:
        paths: Single path or list of paths to scan
        query: Search query string
        extensions: File extensions to include (e.g., ['.py', '.md'])
        top_k: Maximum results to yield (None = all)
        embed_model: Embedding model identifier
        chunk_size: Token size per content chunk
        chunk_overlap: Overlap tokens between consecutive chunks
        threshold: Minimum weighted similarity score to yield
        tokenizer: Optional callable to count tokens (defaults to llama_cpp)
        split_chunks: If True, yield individual chunks; if False, merge adjacent
        includes: Glob patterns to include
        excludes: Glob patterns to exclude
        preprocess: Optional text preprocessing callable
        weights: Similarity weights for name/dir/content components
        batch_size: Texts per embedding request (higher = better RTT amortization)
        max_workers: Concurrent embedding threads (lower = safer for remote)
        show_progress: Display progress bars during embedding
        use_cache: Enable embedding cache to skip unchanged files
        query_prefix: Prefix prepended to query text before embedding
        doc_prefix: Prefix prepended to document text before embedding

    Yields:
        FileSearchResult dictionaries sorted by weighted similarity score
    """

    def default_tokenizer(text):
        return count_tokens(text)

    tokenizer = tokenizer or default_tokenizer

    file_paths, file_names, parent_dirs, chunk_data = collect_file_chunks(
        paths,
        extensions,
        embed_model,
        chunk_size,
        chunk_overlap,
        tokenizer,
        includes,
        excludes,
        show_progress=show_progress,
    )

    logger.debug(f"Parent dirs:\n\n{format_json(parent_dirs)}")
    logger.debug(f"File names:\n\n{format_json(file_names)}")
    logger.debug(f"File paths:\n\n{format_json(file_paths)}")

    if not chunk_data:
        logger.warning("No chunk data found. Returning empty results.")
        return

    unique_files = list(dict.fromkeys(file_paths))
    name_texts = [Path(p).name for p in unique_files]
    dir_texts = [Path(p).parent.name or "root" for p in unique_files]
    chunk_texts = [chunk for _, chunk, _, _, _ in chunk_data]

    # Embed query
    query_processed = preprocess(query) if preprocess else query
    logger.info(f"Embedding query: {query_processed[:100]}...")
    query_vector = embed(
        query_processed,
        model=embed_model,
        return_format="numpy",
        prefix=query_prefix,
    )
    logger.debug(f"Query vector shape: {query_vector.shape}")

    # Embed file names and directory names
    processed_name_texts = [
        preprocess(name) if preprocess else name for name in name_texts
    ]
    processed_dir_texts = [
        preprocess(dir_name) if preprocess else dir_name for dir_name in dir_texts
    ]
    name_dir_texts = processed_name_texts + processed_dir_texts

    if name_dir_texts:
        logger.info(f"Embedding {len(name_dir_texts)} name/dir texts...")
        name_dir_vectors = embed(
            name_dir_texts,
            model=embed_model,
            return_format="numpy",
            batch_size=min(128, len(name_dir_texts)),
            max_workers=max_workers,
            show_progress=True,
            progress_description="Embedding names/dirs",
            prefix=doc_prefix,
        )
        name_vectors = name_dir_vectors[: len(processed_name_texts)]
        dir_vectors = name_dir_vectors[len(processed_name_texts) :]
        logger.debug(f"Name vectors shape: {name_vectors.shape}")
        logger.debug(f"Dir vectors shape: {dir_vectors.shape}")
    else:
        name_vectors = np.array([])
        dir_vectors = np.array([])

    # Embed content chunks
    processed_chunk_texts = [preprocess(c) if preprocess else c for c in chunk_texts]
    logger.info(f"Embedding {len(processed_chunk_texts)} chunks...")
    try:
        chunk_vectors = embed(
            processed_chunk_texts,
            model=embed_model,
            return_format="numpy",
            batch_size=batch_size,
            max_workers=max_workers,
            show_progress=True,
            progress_description="Embedding chunks",
            prefix=doc_prefix,
        )
        if chunk_vectors.size == 0:
            logger.warning("No chunk vectors were embedded. Returning empty results.")
            return
        logger.debug(f"Chunk vectors shape: {chunk_vectors.shape}")
    except Exception as e:
        logger.error(f"Failed to embed chunks: {e}")
        return

    # Compute similarities and yield results
    results: list[FileSearchResult] = []
    chunk_counts = {}
    yielded = 0

    for global_i, content_vector in enumerate(chunk_vectors):
        if global_i >= len(chunk_data):
            break

        file_path, chunk, start_idx, end_idx, num_tokens = chunk_data[global_i]
        file_index = unique_files.index(file_path)

        weighted_sim, name_sim, dir_sim, content_sim = compute_weighted_similarity(
            query_vector,
            name_vectors[file_index],
            dir_vectors[file_index],
            content_vector,
            weights,
        )

        if weighted_sim >= threshold:
            chunk_counts[file_path] = chunk_counts.get(file_path, -1) + 1
            result: FileSearchResult = {
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
                    "num_tokens": num_tokens,
                },
                "text": chunk,
            }
            results.append(result)
            yielded += 1

            if top_k is None or yielded <= top_k:
                yield result

            if top_k is not None and yielded >= top_k:
                logger.info(f"Reached top_k limit ({top_k}). Stopping search.")
                return

    # Sort and optionally merge remaining results
    results.sort(key=lambda x: x["score"], reverse=True)
    for i, r in enumerate(results, 1):
        r["rank"] = i

    if not split_chunks and results:
        logger.info(f"Merging {len(results)} results...")
        merged_results = merge_results(results, tokenizer)
        logger.info(f"Merged into {len(merged_results)} results")
        for i, result in enumerate(
            merged_results if top_k is None else merged_results[:top_k], 1
        ):
            result["rank"] = i
            yield result
