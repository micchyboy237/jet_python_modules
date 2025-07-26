from typing import List, Optional, Set, Union, Tuple, TypedDict
import os
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry
from jet.models.model_types import EmbedModelType

DEFAULT_EMBED_MODEL: EmbedModelType = 'all-MiniLM-L6-v2'
MAX_CONTENT_SIZE = 1000  # Max characters to read from file content


class FileSearchMetadata(TypedDict):
    """Typed dictionary for search result metadata."""
    file_path: str
    start_idx: int
    end_idx: int
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
    model = SentenceTransformerRegistry.load_model(embed_model)
    file_path_obj = Path(file_path)
    file_name = file_path_obj.name.lower()
    # Use 'root' if parent directory is empty or cannot be determined
    parent_dir = file_path_obj.parent.name.lower(
    ) if file_path_obj.parent.name else "root"

    # Encode file name and parent directory
    file_name_vector = model.encode(file_name, convert_to_numpy=True)
    parent_dir_vector = model.encode(parent_dir, convert_to_numpy=True)
    content_vector = None

    # Attempt to read file content if it's a text file
    try:
        if file_path_obj.suffix.lower() in {'.txt', '.py', '.md', '.json', '.csv'}:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(MAX_CONTENT_SIZE).lower()
                if content.strip():
                    content_vector = model.encode(
                        content, convert_to_numpy=True)
    except (UnicodeDecodeError, IOError):
        pass  # Skip content if file is not readable or not text

    return file_name_vector, parent_dir_vector, content_vector


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_product / (norm_a * norm_b) if norm_a * norm_b != 0 else 0.0


def collect_file_chunks(
    paths: Union[str, List[str]],
    extensions: Optional[Set[str]] = None
) -> Tuple[List[str], List[str], List[str], List[Tuple[str, int, int]]]:
    """
    Collect file paths and their text components (names, parent dirs, content) for encoding.

    Args:
        paths: Single directory/file path or list of directory/file paths
        extensions: Optional set of file extensions to filter (e.g., {'.txt', '.py'})

    Returns:
        Tuple of (file_paths, file_names, parent_dirs, contents_with_indices)
        where contents_with_indices is a list of (content, start_idx, end_idx)
    """
    file_paths = []
    file_names = []
    parent_dirs = []
    contents_with_indices = []

    # Normalize input to a list of paths
    path_list = [paths] if isinstance(paths, str) else paths

    for path in path_list:
        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist")

        if os.path.isfile(path):
            # Handle single file
            if extensions and not any(path.lower().endswith(ext) for ext in extensions):
                continue
            file_paths.append(path)
            file_names.append(Path(path).name.lower())
            parent_dirs.append(Path(path).parent.name.lower() or "root")
            # Read content if text file
            content = ""
            start_idx = 0
            end_idx = 0
            try:
                if Path(path).suffix.lower() in {'.txt', '.py', '.md', '.json', '.csv'}:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read(MAX_CONTENT_SIZE).lower()
                        end_idx = min(len(content), MAX_CONTENT_SIZE)
            except (UnicodeDecodeError, IOError):
                pass
            contents_with_indices.append((content, start_idx, end_idx))
        elif os.path.isdir(path):
            # Handle directory
            for root, _, files in os.walk(path):
                for file in files:
                    if extensions and not any(file.lower().endswith(ext) for ext in extensions):
                        continue
                    file_path = os.path.join(root, file)
                    file_paths.append(file_path)
                    file_names.append(file.lower())
                    parent_dirs.append(Path(root).name.lower() or "root")
                    # Read content if text file
                    content = ""
                    start_idx = 0
                    end_idx = 0
                    try:
                        if Path(file_path).suffix.lower() in {'.txt', '.py', '.md', '.json', '.csv'}:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read(MAX_CONTENT_SIZE).lower()
                                end_idx = min(len(content), MAX_CONTENT_SIZE)
                    except (UnicodeDecodeError, IOError):
                        pass
                    contents_with_indices.append((content, start_idx, end_idx))

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
    # Weighted average: 50% file name, 30% directory, 20% content
    weighted_sim = 0.5 * name_sim + 0.3 * dir_sim + 0.2 * content_sim
    return weighted_sim, name_sim, dir_sim, content_sim


def search_files(
    paths: Union[str, List[str]],
    query: str,
    extensions: Optional[Set[str]] = None,
    top_k: int = 5,
    embed_model: EmbedModelType = DEFAULT_EMBED_MODEL
) -> List[FileSearchResult]:
    """
    Search files in directory or file paths using vector similarity with chunked context, filtered by extensions.

    Args:
        paths: Single directory/file path or list of directory/file paths
        query: Search query string
        extensions: Optional set of file extensions to filter (e.g., {'.txt', '.py'})
        top_k: Number of results to return
        embed_model: Embedding model to use

    Returns:
        List of FileSearchResult dictionaries containing rank, score, code, and metadata
    """
    model = SentenceTransformerRegistry.load_model(embed_model)
    query_vector = model.encode(query, convert_to_numpy=True)

    # Collect file chunks
    file_paths, file_names, parent_dirs, contents_with_indices = collect_file_chunks(
        paths, extensions)
    if not file_paths:
        return []

    # Batch encode all components
    name_vectors = model.encode(
        file_names, convert_to_numpy=True, batch_size=32)
    dir_vectors = model.encode(
        parent_dirs, convert_to_numpy=True, batch_size=32)
    contents = [c[0] for c in contents_with_indices]
    content_vectors = model.encode([c for c in contents if c.strip(
    )], convert_to_numpy=True, batch_size=32, show_progress_bar=True) if any(c.strip() for c in contents) else []

    # Calculate similarities
    results: List[FileSearchResult] = []
    content_idx = 0
    for i, (file_path, (content, start_idx, end_idx)) in enumerate(zip(file_paths, contents_with_indices)):
        content_vector = None
        if contents[i].strip() and content_idx < len(content_vectors):
            content_vector = content_vectors[content_idx]
            content_idx += 1
        weighted_sim, name_sim, dir_sim, content_sim = compute_weighted_similarity(
            query_vector, name_vectors[i], dir_vectors[i], content_vector)
        results.append({
            "rank": 0,  # Will be updated after sorting
            "score": weighted_sim,
            "code": content,
            "metadata": {
                "file_path": file_path,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "name_similarity": name_sim,
                "dir_similarity": dir_sim,
                "content_similarity": content_sim
            }
        })

    # Sort by similarity (descending) and assign ranks
    results.sort(key=lambda x: x["score"], reverse=True)
    for i, result in enumerate(results[:top_k], 1):
        result["rank"] = i

    return results[:top_k]
