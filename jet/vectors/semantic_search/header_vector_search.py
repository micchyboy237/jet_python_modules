import re
import uuid
from collections.abc import Callable, Iterator

import numpy as np
from jet.adapters.llama_cpp.chunking_utils import chunk_texts
from jet.adapters.llama_cpp.config import EMBED_MODEL_LG
from jet.adapters.llama_cpp.embed_utils import embed_batch
from jet.adapters.llama_cpp.token_utils import get_tokenizer_fn
from jet.adapters.llama_cpp.types import LLAMACPP_EMBED_KEYS
from jet.code.markdown_types.markdown_parsed_types import HeaderDoc, HeaderSearchResult
from jet.logger import logger
from jet.observability import embedding_span, redact

DEFAULT_EMBED_MODEL: LLAMACPP_EMBED_KEYS = EMBED_MODEL_LG


def preprocess_text(text: str) -> str:
    """Preprocess text by lowercasing and keeping alphanumeric characters."""
    return text.lower()


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))


def collect_header_chunks(
    header_docs: list[HeaderDoc],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    tokenizer_model: LLAMACPP_EMBED_KEYS | None = None,
    buffer: int = 0,
) -> tuple[
    list[int],
    list[str],
    list[str],
    list[tuple[int, str, str, str, str, str, int, int, int]],
]:
    """
    Collect chunked contents for each header along with metadata, preserving original texts.
    """

    def default_tokenizer(text):
        return re.findall(r"\b\w+\b|[^\w\s]", text)

    tokenizer = (
        get_tokenizer_fn(tokenizer_model) if tokenizer_model else default_tokenizer
    )

    doc_indices, headers, headers_context = [], [], []
    contents_with_indices = []

    for header_doc in header_docs:
        doc_index = header_doc["doc_index"]
        original_header = header_doc["header"]
        header = preprocess_text(original_header)

        original_parents = header_doc.get("parent_headers", [])
        parent_text = "\n".join(original_parents) if original_parents else ""
        headers_context_text = (
            f"{parent_text}\n{original_header}" if parent_text else original_header
        )
        headers_context_processed = preprocess_text(headers_context_text)

        if not headers_context_processed:
            logger.warning(f"Empty headers context for doc_index {doc_index}")

        original_content = header_doc["content"]
        doc_indices.append(doc_index)
        headers.append(header)
        headers_context.append(headers_context_processed)

        chunks = chunk_texts(
            original_content, chunk_size, chunk_overlap, tokenizer_model, buffer
        )
        start_idx = 0
        for chunk in chunks:
            if chunk.strip():
                preprocessed_chunk = preprocess_text(chunk)
                end_idx = start_idx + len(chunk)
                num_tokens = tokenizer(chunk)
                if isinstance(num_tokens, list) and len(num_tokens) > 512:
                    chunk = chunk[: int(len(chunk) * 512 / len(num_tokens))]
                    preprocessed_chunk = preprocess_text(chunk)
                    logger.info(
                        f"Truncated content chunk for doc_index {doc_index}, start_idx {start_idx}"
                    )
                contents_with_indices.append(
                    (
                        doc_index,
                        header,
                        preprocessed_chunk,
                        chunk,
                        header,
                        headers_context_processed,
                        start_idx,
                        end_idx,
                        num_tokens,
                    )
                )
                start_idx = max(start_idx, end_idx - chunk_overlap)

    return doc_indices, headers, headers_context, contents_with_indices


def compute_weighted_similarity(
    query_vector: np.ndarray,
    header_vector: np.ndarray,
    parent_vector: np.ndarray,
    content_vector: np.ndarray | None,
    content_tokens: int = 0,
    header_level: int | None = None,
) -> tuple[float, float, float, float]:
    """Compute weighted similarity score based on header, parent, and content components."""
    header_content_sim = (
        cosine_similarity(header_vector, content_vector)
        if content_vector is not None
        else 0.0
    )
    headers_sim = (
        cosine_similarity(query_vector, parent_vector) if np.any(parent_vector) else 0.0
    )
    content_sim = (
        cosine_similarity(query_vector, content_vector)
        if content_vector is not None
        else 0.0
    )

    content_weight = 0.35
    headers_weight = 0.35 if header_level is None or header_level <= 2 else 0.35
    header_content_weight = 0.3

    total = content_weight + headers_weight + header_content_weight
    weighted_sim = (
        header_content_weight * header_content_sim
        + headers_weight * headers_sim
        + content_weight * content_sim
    ) / total

    return weighted_sim, header_content_sim, headers_sim, content_sim


def merge_results(
    results: list[HeaderSearchResult],
    chunk_size: int = 500,
    tokenizer: Callable[[str], int] | None = None,
) -> list[HeaderSearchResult]:
    """Merge adjacent chunks from the same header into a single result."""
    if not results:
        return []

    def default_tokenizer(text):
        return len(re.findall(r"\b\w+\b|[^\w\s]", text))

    tokenizer = tokenizer or default_tokenizer

    grouped: dict[int, list[HeaderSearchResult]] = {}
    for result in results:
        doc_index = result["metadata"]["doc_index"]
        grouped.setdefault(doc_index, []).append(result)

    merged_results: list[HeaderSearchResult] = []

    for doc_index, chunks in grouped.items():
        chunks.sort(key=lambda x: x["metadata"]["start_idx"])
        current_chunk = chunks[0]
        merged_content = current_chunk["content"]
        start_idx = current_chunk["metadata"]["start_idx"]
        end_idx = current_chunk["metadata"]["end_idx"]
        max_score = current_chunk["score"]
        header_content_sim = current_chunk["metadata"]["header_content_similarity"]
        headers_sim = current_chunk["metadata"]["headers_similarity"]
        content_sims = [current_chunk["metadata"]["content_similarity"]]
        chunk_count = 1
        tokens = tokenizer(merged_content)

        preprocessed_header = current_chunk["metadata"]["preprocessed_header"]
        preprocessed_headers_context = current_chunk["metadata"][
            "preprocessed_headers_context"
        ]
        preprocessed_content = current_chunk["metadata"]["preprocessed_content"]
        result_id = current_chunk["id"]

        for next_chunk in chunks[1:]:
            next_start = next_chunk["metadata"]["start_idx"]
            next_end = next_chunk["metadata"]["end_idx"]
            next_content = next_chunk["content"]
            next_preprocessed = next_chunk["metadata"]["preprocessed_content"]

            if next_start <= end_idx:
                overlap = end_idx - next_start
                additional = next_content[overlap:] if overlap > 0 else next_content
                merged_content += additional
                preprocessed_content += (
                    " " + next_preprocessed[overlap:]
                    if overlap > 0
                    else next_preprocessed
                )
                end_idx = max(end_idx, next_end)
                max_score = max(max_score, next_chunk["score"])
                content_sims.append(next_chunk["metadata"]["content_similarity"])
                chunk_count += 1
                tokens = tokenizer(merged_content)
            else:
                avg_sim = sum(content_sims) / chunk_count
                merged_results.append(
                    {
                        "id": result_id,
                        "rank": current_chunk["rank"],
                        "score": max_score,
                        "header": current_chunk["header"],
                        "parent_header": current_chunk["parent_header"],
                        "content": merged_content,
                        "metadata": {
                            "doc_index": doc_index,
                            "doc_id": current_chunk["metadata"]["doc_id"],
                            "level": current_chunk["metadata"]["level"],
                            "parent_level": current_chunk["metadata"]["parent_level"],
                            "parent_headers": current_chunk["metadata"].get(
                                "parent_headers", []
                            ),
                            "start_idx": start_idx,
                            "end_idx": end_idx,
                            "chunk_idx": 0,
                            "header_content_similarity": header_content_sim,
                            "headers_similarity": headers_sim,
                            "content_similarity": avg_sim,
                            "num_tokens": tokens
                            if isinstance(tokens, int)
                            else len(tokens),
                            "preprocessed_header": preprocessed_header,
                            "preprocessed_headers_context": preprocessed_headers_context,
                            "preprocessed_content": preprocessed_content,
                        },
                    }
                )
                current_chunk = next_chunk
                merged_content = current_chunk["content"]
                start_idx = current_chunk["metadata"]["start_idx"]
                end_idx = current_chunk["metadata"]["end_idx"]
                max_score = current_chunk["score"]
                header_content_sim = current_chunk["metadata"][
                    "header_content_similarity"
                ]
                headers_sim = current_chunk["metadata"]["headers_similarity"]
                content_sims = [current_chunk["metadata"]["content_similarity"]]
                chunk_count = 1
                tokens = tokenizer(merged_content)
                preprocessed_header = current_chunk["metadata"]["preprocessed_header"]
                preprocessed_headers_context = current_chunk["metadata"][
                    "preprocessed_headers_context"
                ]
                preprocessed_content = current_chunk["metadata"]["preprocessed_content"]
                result_id = current_chunk["id"]

        avg_sim = sum(content_sims) / chunk_count
        merged_results.append(
            {
                "id": result_id,
                "rank": current_chunk["rank"],
                "score": max_score,
                "header": current_chunk["header"],
                "parent_header": current_chunk["parent_header"],
                "content": merged_content,
                "metadata": {
                    "doc_index": doc_index,
                    "doc_id": current_chunk["metadata"]["doc_id"],
                    "level": current_chunk["metadata"]["level"],
                    "parent_level": current_chunk["metadata"]["parent_level"],
                    "parent_headers": current_chunk["metadata"].get(
                        "parent_headers", []
                    ),
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "chunk_idx": 0,
                    "header_content_similarity": header_content_sim,
                    "headers_similarity": headers_sim,
                    "content_similarity": avg_sim,
                    "num_tokens": tokens if isinstance(tokens, int) else len(tokens),
                    "preprocessed_header": preprocessed_header,
                    "preprocessed_headers_context": preprocessed_headers_context,
                    "preprocessed_content": preprocessed_content,
                },
            }
        )

    merged_results.sort(key=lambda x: x["score"], reverse=True)
    for i, result in enumerate(merged_results, 1):
        result["rank"] = i
    return merged_results


def search_headers(
    header_docs: list[HeaderDoc],
    query: str,
    top_k: int | None = None,
    embed_model: str = DEFAULT_EMBED_MODEL,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    buffer: int = 0,
    threshold: float = 0.0,
    tokenizer_model: LLAMACPP_EMBED_KEYS | None = None,
    merge_chunks: bool = True,
) -> Iterator[HeaderSearchResult]:
    """
    Search headers using vector similarity with optimized batch embedding.
    Uses jet.adapters.llama_cpp.embed_utils.embed_batch for parallel,
    deduplicated, and observable embedding generation.
    """

    def default_tokenizer(text):
        return len(re.findall(r"\b\w+\b|[^\w\s]", text))

    tokenizer = (
        get_tokenizer_fn(tokenizer_model) if tokenizer_model else default_tokenizer
    )

    total_tokens = sum(len(tokenizer(doc["content"])) for doc in header_docs)
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

    all_texts = [query_processed] + header_texts + parent_texts + chunked_texts

    # Use optimized embed_batch with observability span
    with embedding_span(
        name="search_headers.embed_all",
        model_name=embed_model,
        texts=[redact(t[:200]) for t in all_texts],
    ) as emb_span:
        all_vectors = embed_batch(
            texts=all_texts,
            model=embed_model,
            max_workers=6,
            show_progress=True,
            return_format="numpy",
            batch_size=64,
            progress_description="Embedding headers & chunks",
        )
        emb_span.set_attribute("embedding.total_texts", len(all_texts))
        emb_span.set_attribute("embedding.unique_docs", len(unique_docs))
        emb_span.set_attribute("embedding.chunk_count", len(chunked_texts))

    query_vector = all_vectors[0]
    num_headers = len(header_texts)
    num_parents = len(parent_texts)

    header_vectors = all_vectors[1 : num_headers + 1]
    parent_vectors = all_vectors[num_headers + 1 : num_headers + 1 + num_parents]
    content_vectors = all_vectors[num_headers + 1 + num_parents :]

    results: list[HeaderSearchResult] = []
    chunk_counts: dict[int, int] = {}

    for i, (
        doc_index,
        header,
        chunk,
        original_chunk,
        preprocessed_header,
        preprocessed_headers_context,
        start_idx,
        end_idx,
        num_tokens,
    ) in enumerate(chunk_data):
        unique_doc_idx = unique_docs.index(doc_index)
        header_doc = next(hd for hd in header_docs if hd["doc_index"] == doc_index)
        content_vector = content_vectors[i]

        token_count = num_tokens if isinstance(num_tokens, int) else len(num_tokens)
        weighted_sim, hc_sim, h_sim, c_sim = compute_weighted_similarity(
            query_vector,
            header_vectors[unique_doc_idx],
            parent_vectors[unique_doc_idx],
            content_vector,
            token_count,
            header_doc["level"],
        )

        if weighted_sim >= threshold:
            chunk_counts[doc_index] = chunk_counts.get(doc_index, -1) + 1
            results.append(
                {
                    "id": str(uuid.uuid4()),
                    "rank": 0,
                    "score": float(weighted_sim),
                    "header": header_doc["header"],
                    "parent_header": header_doc["parent_header"],
                    "content": original_chunk,
                    "metadata": {
                        "doc_index": doc_index,
                        "doc_id": header_doc["id"],
                        "level": header_doc["level"],
                        "parent_level": header_doc["parent_level"],
                        "parent_headers": header_doc.get("parent_headers", []),
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "chunk_idx": chunk_counts[doc_index],
                        "source": header_doc["source"],
                        "header_content_similarity": float(hc_sim),
                        "headers_similarity": float(h_sim),
                        "content_similarity": float(c_sim),
                        "num_tokens": token_count,
                        "preprocessed_header": preprocessed_header,
                        "preprocessed_headers_context": preprocessed_headers_context,
                        "preprocessed_content": chunk,
                    },
                }
            )

    results.sort(key=lambda x: x["score"], reverse=True)

    if merge_chunks:
        results = merge_results(results, dynamic_chunk_size, tokenizer)

    final_results = results if top_k is None else results[:top_k]
    for i, result in enumerate(final_results, 1):
        result["rank"] = i
        yield result
