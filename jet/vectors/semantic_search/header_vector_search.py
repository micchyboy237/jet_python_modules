import uuid
from jet.libs.llama_cpp.embeddings import LlamacppEmbedding
from jet.llm.models import OLLAMA_EMBED_MODELS
from jet.models.model_types import ModelType
from jet.models.tokenizer.base import get_tokenizer_fn
from jet.models.utils import get_context_size
from jet.wordnet.text_chunker import chunk_texts
import re
from typing import List, Optional, Tuple, Iterator, Callable
import numpy as np
from jet.logger import logger
# from jet.models.embeddings.base import generate_embeddings
# from jet.llm.utils.embeddings import generate_embeddings
from jet.code.markdown_types.markdown_parsed_types import HeaderDoc, HeaderSearchResult

DEFAULT_EMBED_MODEL: OLLAMA_EMBED_MODELS = "all-minilm:33m"
MAX_CONTENT_SIZE = 1000


# def preprocess_text(
#     text: str,
# ) -> str:
#     if not text or not text.strip():
#         return ""
#     # Remove subsequent newlines and leading/trailing newlines
#     text = re.sub(r'\n\s*\n+', '\n', text)
#     preserve_chars = {'-', '_'}
#     pattern = r'[^a-z0-9\s' + ''.join(map(re.escape, preserve_chars)) + r']'
#     processed_lines = []
#     for line in text.splitlines():
#         # Expand contractions
#         for contraction, expanded in TEXT_CONTRACTIONS_EN.items():
#             line = re.sub(r'\b' + re.escape(contraction) + r'\b',
#                           expanded, line, flags=re.IGNORECASE)
#         # Lowercase
#         line = line.lower()
#         # Remove unwanted characters but preserve allowed ones
#         line = re.sub(pattern, '', line)
#         # Normalize whitespace
#         line = re.sub(r'\s+', ' ', line.strip())
#         if line:
#             processed_lines.append(line)
#     return '\n'.join(processed_lines)

def preprocess_text(text: str) -> str:
    """Preprocess text by lowercasing and keeping alphanumeric characters."""
    # text = text.lower()
    # text = ''.join(char for char in text if char.isalnum() or char.isspace())
    return text


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_product / (norm_a * norm_b) if norm_a * norm_b != 0 else 0.0


def collect_header_chunks(
    header_docs: List[HeaderDoc],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    tokenizer_model: Optional[ModelType] = None,
    buffer: int = 0,
) -> Tuple[List[int], List[str], List[str], List[Tuple[int, str, str, str, str, str, int, int, int]]]:
    """
    Collect chunked contents for each header along with metadata, preserving original texts.
    Args:
        header_docs: List of HeaderDoc objects to process
        chunk_size: Size of content chunks
        chunk_overlap: Overlap between chunks
        tokenizer_model: Optional LLM model name for token-based chunking
    Returns:
        Tuple of (doc_indices, headers, headers_context, contents_with_indices)
        where contents_with_indices = List of (doc_index, header, content_chunk, original_content_chunk, preprocessed_header, preprocessed_headers_context, start_idx, end_idx, num_tokens)
    """
    def default_tokenizer(text): return re.findall(r'\b\w+\b|[^\w\s]', text)
    tokenizer = get_tokenizer_fn(
        tokenizer_model) if tokenizer_model else default_tokenizer

    context_size = get_context_size(
        tokenizer_model) if tokenizer_model else 512
    doc_indices, headers, headers_context = [], [], []
    contents_with_indices = []

    for header_doc in header_docs:
        doc_index = header_doc['doc_index']
        original_header = header_doc['header']
        header = preprocess_text(original_header)
        header_buffer = buffer or len(tokenizer(original_header))
        original_parents = header_doc.get('parent_headers', [])
        parent_text = '\n'.join(original_parents) if original_parents else ''
        headers_context_text = f"{parent_text}\n{original_header}" if parent_text else original_header
        headers_context_processed = preprocess_text(
            headers_context_text) if headers_context_text else ""
        if not headers_context_processed:
            logger.warning(f"Empty headers context for doc_index {doc_index}")

        original_content = header_doc['content']
        doc_indices.append(doc_index)
        headers.append(header)
        headers_context.append(headers_context_processed)

        # Use chunk_texts from text_chunker
        chunks = chunk_texts(original_content, chunk_size,
                             chunk_overlap, tokenizer_model, header_buffer)

        start_idx = 0
        for chunk in chunks:
            if chunk.strip():
                preprocessed_chunk = preprocess_text(chunk)
                end_idx = start_idx + len(chunk)
                num_tokens = tokenizer(chunk)
                if len(num_tokens) > context_size:
                    chunk = chunk[:context_size]
                    preprocessed_chunk = preprocess_text(chunk)
                    logger.info(
                        f"Truncated content chunk to context_size tokens for doc_index {doc_index}, start_idx {start_idx}")
                contents_with_indices.append(
                    (doc_index, header, preprocessed_chunk, chunk, header, headers_context_processed, start_idx, end_idx, num_tokens))
                start_idx = end_idx - chunk_overlap if end_idx - \
                    start_idx > chunk_overlap else start_idx
    return doc_indices, headers, headers_context, contents_with_indices


def compute_weighted_similarity(
    query_vector: np.ndarray,
    header_vector: np.ndarray,
    parent_vector: np.ndarray,
    content_vector: Optional[np.ndarray],
    content_tokens: int = 0,
    header_level: Optional[int] = None
) -> Tuple[float, float, float, float]:
    """
    Compute weighted similarity score and individual scores for a header based on its components.
    Args:
        query_vector: Encoded query vector
        header_vector: Encoded header vector
        parent_vector: Encoded concatenated headers context vector
        content_vector: Encoded content vector (if available)
        content_tokens: Number of tokens in the content chunk
        header_level: Level of the header (for dynamic weighting)
    Returns:
        Tuple of (weighted_similarity, header_content_similarity, headers_similarity, content_similarity)
    """
    header_content_sim = cosine_similarity(
        header_vector, content_vector) if content_vector is not None else 0.0
    headers_sim = cosine_similarity(
        query_vector, parent_vector) if np.any(parent_vector) else 0.0
    content_sim = 0.0
    if content_vector is not None:
        content_sim = cosine_similarity(query_vector, content_vector)
    content_weight = 0.35 if content_tokens >= 100 else 0.35
    headers_weight = 0.35 if header_level is None or header_level <= 2 else 0.35
    header_content_weight = 0.3 if content_tokens >= 100 else 0.3
    total = content_weight + headers_weight + header_content_weight
    content_weight /= total
    headers_weight /= total
    header_content_weight /= total
    weighted_sim = header_content_weight * header_content_sim + \
        headers_weight * headers_sim + content_weight * content_sim
    return weighted_sim, header_content_sim, headers_sim, content_sim


def merge_results(
    results: List[HeaderSearchResult],
    chunk_size: int = 500,
    tokenizer: Optional[Callable[[str], int]] = None
) -> List[HeaderSearchResult]:
    """
    Merge adjacent chunks from the same header into a single result, preserving order and metadata.
    Args:
        results: List of HeaderSearchResult dictionaries, potentially containing adjacent chunks.
        tokenizer: Optional callable to count tokens in text. Defaults to regex-based word and punctuation counting.
    Returns:
        List of HeaderSearchResult dictionaries with adjacent chunks merged.
    """
    if not results:
        return []
    def default_tokenizer(text): return len(
        re.findall(r'\b\w+\b|[^\w\s]', text))
    tokenizer = tokenizer or default_tokenizer
    grouped: dict[int, List[HeaderSearchResult]] = {}
    for result in results:
        doc_index = result["metadata"]["doc_index"]
        if doc_index not in grouped:
            grouped[doc_index] = []
        grouped[doc_index].append(result)
    merged_results: List[HeaderSearchResult] = []
    for doc_index, chunks in grouped.items():
        chunks.sort(key=lambda x: x["metadata"]["start_idx"])
        current_chunk = chunks[0]
        merged_content = current_chunk["content"]
        start_idx = current_chunk["metadata"]["start_idx"]
        end_idx = current_chunk["metadata"]["end_idx"]
        max_score = current_chunk["score"]  # Track max score instead of total
        header_content_sim = current_chunk["metadata"]["header_content_similarity"]
        headers_sim = current_chunk["metadata"]["headers_similarity"]
        content_sims = [current_chunk["metadata"]["content_similarity"]]
        chunk_count = 1
        tokens = tokenizer(merged_content)
        preprocessed_header = current_chunk["metadata"]["preprocessed_header"]
        preprocessed_headers_context = current_chunk["metadata"]["preprocessed_headers_context"]
        preprocessed_content = current_chunk["metadata"]["preprocessed_content"]
        result_id = current_chunk["id"]  # Preserve the original result ID
        for next_chunk in chunks[1:]:
            next_start = next_chunk["metadata"]["start_idx"]
            next_end = next_chunk["metadata"]["end_idx"]
            next_content = next_chunk["content"]
            next_preprocessed_content = next_chunk["metadata"]["preprocessed_content"]
            if next_start <= end_idx:
                new_end = max(end_idx, next_end)
                overlap = end_idx - next_start
                additional_content = next_content[overlap:
                                                  ] if overlap > 0 else next_content
                merged_content += additional_content
                preprocessed_content += " " + \
                    next_preprocessed_content[overlap:] if overlap > 0 else next_preprocessed_content
                end_idx = new_end
                max_score = max(max_score, next_chunk["score"])  # Update max score
                content_sims.append(
                    next_chunk["metadata"]["content_similarity"])
                chunk_count += 1
                tokens = tokenizer(merged_content)
            else:
                avg_content_sim = sum(content_sims) / chunk_count
                merged_results.append({
                    "id": result_id,
                    "rank": current_chunk["rank"],
                    "score": max_score,  # Use max score
                    "header": current_chunk["header"],
                    "parent_header": current_chunk["parent_header"],
                    "content": merged_content,
                    "metadata": {
                        "doc_index": doc_index,
                        "doc_id": current_chunk["metadata"]["doc_id"],
                        "level": current_chunk["metadata"]["level"],
                        "parent_level": current_chunk["metadata"]["parent_level"],
                        "parent_headers": current_chunk["metadata"].get("parent_headers", []),
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "chunk_idx": 0,
                        "header_content_similarity": header_content_sim,
                        "headers_similarity": headers_sim,
                        "content_similarity": avg_content_sim,
                        "num_tokens": tokens if isinstance(tokens, int) else len(tokens),
                        "preprocessed_header": preprocessed_header,
                        "preprocessed_headers_context": preprocessed_headers_context,
                        "preprocessed_content": preprocessed_content
                    },
                })
                current_chunk = next_chunk
                merged_content = current_chunk["content"]
                start_idx = current_chunk["metadata"]["start_idx"]
                end_idx = current_chunk["metadata"]["end_idx"]
                max_score = current_chunk["score"]  # Reset to new chunk's score
                header_content_sim = current_chunk["metadata"]["header_content_similarity"]
                headers_sim = current_chunk["metadata"]["headers_similarity"]
                content_sims = [current_chunk["metadata"]
                                ["content_similarity"]]
                chunk_count = 1
                tokens = tokenizer(merged_content)
                preprocessed_header = current_chunk["metadata"]["preprocessed_header"]
                preprocessed_headers_context = current_chunk["metadata"]["preprocessed_headers_context"]
                preprocessed_content = current_chunk["metadata"]["preprocessed_content"]
                result_id = current_chunk["id"]
        avg_content_sim = sum(content_sims) / chunk_count
        merged_results.append({
            "id": result_id,
            "rank": current_chunk["rank"],
            "score": max_score,  # Use max score
            "header": current_chunk["header"],
            "parent_header": current_chunk["parent_header"],
            "content": merged_content,
            "metadata": {
                "doc_index": doc_index,
                "doc_id": current_chunk["metadata"]["doc_id"],
                "level": current_chunk["metadata"]["level"],
                "parent_level": current_chunk["metadata"]["parent_level"],
                "parent_headers": current_chunk["metadata"].get("parent_headers", []),
                "start_idx": start_idx,
                "end_idx": end_idx,
                "chunk_idx": 0,
                "header_content_similarity": header_content_sim,
                "headers_similarity": headers_sim,
                "content_similarity": avg_content_sim,
                "num_tokens": tokens if isinstance(tokens, int) else len(tokens),
                "preprocessed_header": preprocessed_header,
                "preprocessed_headers_context": preprocessed_headers_context,
                "preprocessed_content": preprocessed_content
            },
        })
    merged_results.sort(key=lambda x: x["score"], reverse=True)
    for i, result in enumerate(merged_results, 1):
        result["rank"] = i
    return merged_results


def search_headers(
    header_docs: List[HeaderDoc],
    query: str,
    top_k: Optional[int] = None,
    embed_model: str = DEFAULT_EMBED_MODEL,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    buffer: int = 0,
    threshold: float = 0.0,
    tokenizer_model: Optional[ModelType] = None,
    merge_chunks: bool = True  # Enable merging by default
) -> Iterator[HeaderSearchResult]:
    """
    Search headers using vector similarity on chunked contents + header metadata.
    Yields up to top_k results iteratively that meet the threshold, preserving original texts.
    """
    def default_tokenizer(text): return len(re.findall(r'\b\w+\b|[^\w\s]', text))
    tokenizer = get_tokenizer_fn(tokenizer_model) if tokenizer_model else default_tokenizer
    
    # Dynamically adjust chunk size based on content length
    total_tokens = sum(len(tokenizer(doc['content'])) for doc in header_docs)
    avg_doc_tokens = total_tokens / len(header_docs) if header_docs else chunk_size
    dynamic_chunk_size = min(chunk_size, max(128, int(avg_doc_tokens / 2)))

    query_processed = preprocess_text(query)
    doc_indices, headers, headers_context, chunk_data = collect_header_chunks(
        header_docs, dynamic_chunk_size, chunk_overlap, tokenizer_model, buffer
    )
    if not chunk_data:
        logger.debug("No chunk data available, returning empty iterator")
        return

    unique_docs = list(dict.fromkeys(doc_indices))
    header_texts = [headers[doc_indices.index(idx)] for idx in unique_docs]
    parent_texts = [headers_context[doc_indices.index(idx)] for idx in unique_docs]
    chunked_texts = [chunk for _, _, chunk, _, _, _, _, _, _ in chunk_data]

    logger.debug(
        f"Text counts: query=1, headers={len(header_texts)}, parents={len(parent_texts)}, chunks={len(chunked_texts)}"
    )
    all_texts = [query_processed] + header_texts + parent_texts + chunked_texts
    logger.info(
        f"Generating embeddings for {len(all_texts)} texts:\n"
        f"  1 query\n"
        f"  {len(header_texts)} headers\n"
        f"  {len(parent_texts)} headers context\n"
        f"  {len(chunked_texts)} chunks"
    )

    client = LlamacppEmbedding(model=embed_model)
    all_vectors = client.get_embeddings(
        all_texts,
        return_format="numpy",
        batch_size=32,
        show_progress=True
    )

    query_vector = all_vectors[0]
    num_headers = len(header_texts)
    num_parents = len(parent_texts)
    header_vectors = all_vectors[1:num_headers + 1]
    parent_vectors = all_vectors[num_headers + 1:num_headers + 1 + num_parents]
    content_vectors = all_vectors[num_headers + 1 + num_parents:]

    results: List[HeaderSearchResult] = []
    chunk_counts = {}
    for i, (doc_index, header, chunk, original_chunk, preprocessed_header, preprocessed_headers_context, start_idx, end_idx, num_tokens) in enumerate(chunk_data):
        unique_doc_idx = unique_docs.index(doc_index)
        header_doc = next(hd for hd in header_docs if hd['doc_index'] == doc_index)
        content_vector = content_vectors[i]

        # Adjust weights for better relevance
        content_weight = 0.5 if len(num_tokens) >= 100 else 0.3
        headers_weight = 0.3 if header_doc['level'] <= 2 else 0.2
        header_content_weight = 0.2
        total = content_weight + headers_weight + header_content_weight
        content_weight /= total
        headers_weight /= total
        header_content_weight /= total

        weighted_sim, header_content_sim, headers_sim, content_sim = compute_weighted_similarity(
            query_vector, header_vectors[unique_doc_idx], parent_vectors[unique_doc_idx],
            content_vector, len(num_tokens), header_doc['level']
        )

        if weighted_sim >= threshold:
            chunk_counts[doc_index] = chunk_counts.get(doc_index, -1) + 1
            result: HeaderSearchResult = {
                "id": str(uuid.uuid4()),
                "rank": 0,
                "score": float(weighted_sim),
                "header": header_doc['header'],
                "parent_header": header_doc['parent_header'],
                "content": original_chunk,
                "metadata": {
                    "doc_index": doc_index,
                    "doc_id": header_doc['id'],
                    "level": header_doc['level'],
                    "parent_level": header_doc['parent_level'],
                    "parent_headers": header_doc.get('parent_headers', []),
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "chunk_idx": chunk_counts[doc_index],
                    "source": header_doc['source'],
                    "header_content_similarity": float(header_content_sim),
                    "headers_similarity": float(headers_sim),
                    "content_similarity": float(content_sim),
                    "num_tokens": num_tokens if isinstance(num_tokens, int) else len(num_tokens),
                    "preprocessed_header": preprocessed_header,
                    "preprocessed_headers_context": preprocessed_headers_context,
                    "preprocessed_content": chunk
                },
            }
            results.append(result)

    results.sort(key=lambda x: x["score"], reverse=True)
    if merge_chunks:
        results = merge_results(results, dynamic_chunk_size, tokenizer)
    
    for i, result in enumerate(results if top_k is None else results[:top_k], 1):
        result["rank"] = i
        yield result
