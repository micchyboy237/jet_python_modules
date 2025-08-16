import os
import re
import nltk
import fnmatch
import numpy as np
from nltk.stem import PorterStemmer
from typing import List, Optional, Union, Tuple, TypedDict, Iterator, Callable
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pathlib import Path
from jet.models.embeddings.base import generate_embeddings
from jet.models.model_registry.transformers.cross_encoder_model_registry import CrossEncoderRegistry
from jet.models.model_types import EmbedModelType, RerankModelType
from jet.logger import logger
from jet.wordnet.text_chunker import chunk_texts_with_data

DEFAULT_EMBED_MODEL: EmbedModelType = 'static-retrieval-mrl-en-v1'
DEFAULT_RERANK_MODEL: RerankModelType = 'ms-marco-MiniLM-L6-v2'

# Download NLTK data for BM25 tokenization
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


class FileSearchMetadata(TypedDict):
    """Typed dictionary for search result metadata."""
    file_path: str
    start_idx: int
    end_idx: int
    chunk_idx: int
    name_similarity: float
    dir_similarity: float
    content_similarity: float
    metadata_similarity: float
    cross_encoder_score: float
    bm25_score: float
    num_tokens: int


class FileSearchResult(TypedDict):
    """Typed dictionary for search result structure."""
    rank: int
    score: float
    metadata: FileSearchMetadata
    text: str  # Changed from 'code' to 'text'


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_product / (norm_a * norm_b) if norm_a * norm_b != 0 else 0.0


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


def collect_file_chunks(
    paths: Union[str, List[str]],
    extensions: List[str] = None,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    tokenizer: Optional[Callable[[str], int]] = None,
    includes: Optional[List[str]] = None,
    excludes: Optional[List[str]] = None
) -> Tuple[List[str], List[str], List[str], List[Tuple[str, str, int, int, int]]]:
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
    def default_tokenizer(text): return len(
        re.findall(r'\b\w+\b|[^\w\s]', text))
    tokenizer = tokenizer or default_tokenizer
    file_paths = get_matched_files(paths, extensions, includes, excludes)
    file_names = []
    parent_dirs = []
    contents_with_indices = []

    for file_path in file_paths:
        file_path_obj = Path(file_path)
        file_names.append(file_path_obj.name)
        parent_dirs.append(file_path_obj.parent.name or "root")
        try:
            if file_path_obj.suffix in {'.txt', '.py', '.md', '.json', '.csv'}:
                with open(file_path, 'r', encoding='utf-8') as f:
                    full_content = f.read()
                # Use chunk_texts_with_data for chunking
                chunks = chunk_texts_with_data(
                    texts=[full_content],
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    model=None,  # Using default word-based tokenization
                    doc_ids=[file_path],  # Use file_path as doc_id
                    buffer=0
                )
                # Map ChunkResult to contents_with_indices format
                for chunk in chunks:
                    contents_with_indices.append(
                        (
                            file_path,
                            chunk['content'],
                            chunk['start_idx'],
                            chunk['end_idx'],
                            chunk['num_tokens']
                        )
                    )
        except (UnicodeDecodeError, IOError):
            continue

    return file_paths, file_names, parent_dirs, contents_with_indices


def compute_dynamic_weights(
    query: str,
    file_path: str,
    content: str,
    stop_words: set = set(stopwords.words('english'))
) -> Tuple[float, float, float, float]:
    """
    Compute dynamic weights for similarity scores based on query and document characteristics.
    Args:
        query: Search query string
        file_path: File path for metadata
        content: Content chunk
        stop_words: Set of stop words for keyword analysis
    Returns:
        Tuple of (name_weight, dir_weight, content_weight, metadata_weight)
    """
    ps = PorterStemmer()
    query_tokens = set(word_tokenize(query.lower())) - stop_words
    stemmed_query_tokens = {ps.stem(token) for token in query_tokens}
    file_path_obj = Path(file_path)
    file_name = file_path_obj.stem.lower()
    dir_name = file_path_obj.parent.name.lower()
    stemmed_file_name = {
        ps.stem(word) for word in file_name.replace('_', ' ').split() if word}
    stemmed_dir_name = {ps.stem(word)
                        for word in dir_name.replace('_', ' ').split() if word}

    # Match if stemmed tokens overlap or if short tokens are likely abbreviations
    def is_likely_abbreviation(short_token: str, long_token: str) -> bool:
        if len(short_token) > 3:
            return False
        # Check if short_token shares at least two initial characters or is 'ml' for machine learning
        return (short_token == 'ml' and long_token in {'machin', 'learn'}) or (
            long_token.startswith(short_token[:2]) and len(short_token) >= 2
        )

    dir_weight = 0.3 if (
        any(stem in stemmed_dir_name for stem in stemmed_query_tokens) or
        any(is_likely_abbreviation(stem_dir, stem_query)
            for stem_dir in stemmed_dir_name for stem_query in stemmed_query_tokens)
    ) else 0.2
    name_weight = 0.4 if (
        any(stem in stemmed_file_name for stem in stemmed_query_tokens) or
        any(is_likely_abbreviation(stem_file, stem_query)
            for stem_file in stemmed_file_name for stem_query in stemmed_query_tokens)
    ) else 0.3
    content_weight = 0.5 if len(content.split()) < 200 else 0.4
    metadata_weight = 0.1

    total = name_weight + dir_weight + content_weight + metadata_weight
    return (
        name_weight / total,
        dir_weight / total,
        content_weight / total,
        metadata_weight / total
    )


def compute_hybrid_similarity(
    query: str,
    query_vector: np.ndarray,
    name_vector: np.ndarray,
    dir_vector: np.ndarray,
    content_vector: np.ndarray,
    metadata_vector: np.ndarray,
    tokenized_corpus: List[List[str]],
    corpus_index: int,
    cross_encoder: CrossEncoder,
    file_path: str,
    content: str
) -> Tuple[float, float, float, float, float, float]:
    name_sim = cosine_similarity(query_vector, name_vector)
    dir_sim = cosine_similarity(query_vector, dir_vector)
    content_sim = cosine_similarity(query_vector, content_vector)
    metadata_sim = cosine_similarity(query_vector, metadata_vector)

    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = word_tokenize(query.lower())
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_range = max(bm25_scores) - min(bm25_scores)
    bm25_score = (bm25_scores[corpus_index] - min(bm25_scores)) / \
        (bm25_range + 1e-10) if bm25_range > 0 else 0.0

    raw_cross_encoder_score = cross_encoder.predict([(query, content)])[0]
    # Normalize cross-encoder score to [0, 1] using sigmoid
    cross_encoder_score = 1 / (1 + np.exp(-raw_cross_encoder_score))

    name_weight, dir_weight, content_weight, metadata_weight = compute_dynamic_weights(
        query, file_path, content)

    weighted_sim = (
        name_weight * name_sim +
        dir_weight * dir_sim +
        content_weight * content_sim +
        metadata_weight * metadata_sim +
        0.2 * cross_encoder_score +
        0.1 * bm25_score
    )
    return weighted_sim, name_sim, dir_sim, content_sim, metadata_sim, cross_encoder_score, bm25_score


def merge_results(
    results: List[FileSearchResult],
    tokenizer: Optional[Callable[[str], int]] = None
) -> List[FileSearchResult]:
    """
    Merge adjacent chunks from the same file into a single result, preserving order and metadata.
    Uses the largest score among chunks for the merged result.
    Args:
        results: List of FileSearchResult dictionaries, potentially containing adjacent chunks.
        tokenizer: Optional callable to count tokens in text.
    Returns:
        List of FileSearchResult dictionaries with adjacent chunks merged.
    """
    if not results:
        return []

    def default_tokenizer(text): return len(
        re.findall(r'\b\w+\b|[^\w\s]', text))
    tokenizer = tokenizer or default_tokenizer

    grouped: dict[str, List[FileSearchResult]] = {}
    for result in results:
        file_path = result["metadata"]["file_path"]
        if file_path not in grouped:
            grouped[file_path] = []
        grouped[file_path].append(result)

    merged_results: List[FileSearchResult] = []
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
        metadata_sims = [current_chunk["metadata"]["metadata_similarity"]]
        cross_encoder_sims = [current_chunk["metadata"]["cross_encoder_score"]]
        bm25_sims = [current_chunk["metadata"]["bm25_score"]]
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
                content_sims.append(
                    next_chunk["metadata"]["content_similarity"])
                metadata_sims.append(
                    next_chunk["metadata"]["metadata_similarity"])
                cross_encoder_sims.append(
                    next_chunk["metadata"]["cross_encoder_score"])
                bm25_sims.append(next_chunk["metadata"]["bm25_score"])
                chunk_count += 1
                tokens = tokenizer(merged_text)
            else:
                avg_content_sim = sum(content_sims) / chunk_count
                avg_metadata_sim = sum(metadata_sims) / chunk_count
                avg_cross_encoder_sim = sum(cross_encoder_sims) / chunk_count
                avg_bm25_sim = sum(bm25_sims) / chunk_count
                merged_results.append({
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
                        "metadata_similarity": avg_metadata_sim,
                        "cross_encoder_score": avg_cross_encoder_sim,
                        "bm25_score": avg_bm25_sim,
                        "num_tokens": tokens
                    },
                    "text": merged_text,
                })
                current_chunk = next_chunk
                merged_text = current_chunk["text"]
                start_idx = current_chunk["metadata"]["start_idx"]
                end_idx = current_chunk["metadata"]["end_idx"]
                max_score = current_chunk["score"]
                name_sim = current_chunk["metadata"]["name_similarity"]
                dir_sim = current_chunk["metadata"]["dir_similarity"]
                content_sims = [current_chunk["metadata"]
                                ["content_similarity"]]
                metadata_sims = [current_chunk["metadata"]
                                 ["metadata_similarity"]]
                cross_encoder_sims = [
                    current_chunk["metadata"]["cross_encoder_score"]]
                bm25_sims = [current_chunk["metadata"]["bm25_score"]]
                chunk_count = 1
                tokens = tokenizer(merged_text)

        avg_content_sim = sum(content_sims) / chunk_count
        avg_metadata_sim = sum(metadata_sims) / chunk_count
        avg_cross_encoder_sim = sum(cross_encoder_sims) / chunk_count
        avg_bm25_sim = sum(bm25_sims) / chunk_count
        merged_results.append({
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
                "metadata_similarity": avg_metadata_sim,
                "cross_encoder_score": avg_cross_encoder_sim,
                "bm25_score": avg_bm25_sim,
                "num_tokens": tokens
            },
            "text": merged_text,
        })

    merged_results.sort(key=lambda x: x["score"], reverse=True)
    for i, result in enumerate(merged_results, 1):
        result["rank"] = i
    return merged_results


def search_files(
    paths: Union[str, List[str]],
    query: str,
    extensions: Optional[List[str]] = None,
    top_k: Optional[int] = None,
    embed_model: EmbedModelType = DEFAULT_EMBED_MODEL,
    rerank_model: RerankModelType = DEFAULT_RERANK_MODEL,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    threshold: float = 0.0,
    tokenizer: Optional[Callable[[str], int]] = None,
    split_chunks: bool = False,
    includes: Optional[List[str]] = None,
    excludes: Optional[List[str]] = None
) -> Iterator[FileSearchResult]:
    def default_tokenizer(text): return len(
        re.findall(r'\b\w+\b|[^\w\s]', text))
    tokenizer = tokenizer or default_tokenizer
    cross_encoder = CrossEncoderRegistry.load_model(rerank_model)
    file_paths, file_names, parent_dirs, chunk_data = collect_file_chunks(
        paths, extensions, chunk_size, chunk_overlap, tokenizer, includes, excludes)
    if not chunk_data:
        return
    unique_files = list(dict.fromkeys(file_paths))
    metadata_texts = [f"{Path(p).suffix} {Path(p).name}" for p in unique_files]
    tokenized_corpus = [word_tokenize(chunk.lower())
                        for _, chunk, _, _, _ in chunk_data]
    name_texts = [Path(p).name for p in unique_files]
    dir_texts = [Path(p).parent.name or "root" for p in unique_files]
    chunk_texts = [chunk for _, chunk, _, _, _ in chunk_data]
    logger.info(
        f"Generating embeddings for {len(name_texts + dir_texts + chunk_texts + metadata_texts) + 1} texts:\n"
        f"  1 query\n"
        f"  {len(name_texts)} names\n"
        f"  {len(dir_texts)} dirs\n"
        f"  {len(chunk_texts)} chunks\n"
        f"  {len(metadata_texts)} metadata"
    )
    all_texts = [query] + name_texts + dir_texts + chunk_texts + metadata_texts
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
    num_chunks = len(chunk_texts)
    name_vectors = all_vectors[1:num_names + 1]
    dir_vectors = all_vectors[num_names + 1:num_names + 1 + num_dirs]
    content_vectors = all_vectors[num_names + 1 +
                                  num_dirs:num_names + 1 + num_dirs + num_chunks]
    metadata_vectors = all_vectors[num_names + 1 + num_dirs + num_chunks:]
    results: List[FileSearchResult] = []
    chunk_counts = {}
    for i, (file_path, chunk, start_idx, end_idx, num_tokens) in enumerate(chunk_data):
        file_index = unique_files.index(file_path)
        weighted_sim, name_sim, dir_sim, content_sim, metadata_sim, cross_encoder_score, bm25_score = compute_hybrid_similarity(
            query, query_vector, name_vectors[file_index], dir_vectors[file_index],
            content_vectors[i], metadata_vectors[file_index], tokenized_corpus, i,
            cross_encoder, file_path, chunk
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
                    "metadata_similarity": float(metadata_sim),
                    "cross_encoder_score": float(cross_encoder_score),
                    "bm25_score": float(bm25_score),
                    "num_tokens": num_tokens
                },
                "text": chunk,  # Changed from 'code' to 'text'
            }
            results.append(result)
    results.sort(key=lambda x: x["score"], reverse=True)
    if not split_chunks:
        results = merge_results(results, tokenizer)
    for i, result in enumerate(results if top_k is None else results[:top_k], 1):
        result["rank"] = i
        yield result
