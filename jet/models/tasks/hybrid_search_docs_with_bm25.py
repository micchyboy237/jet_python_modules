import string
from typing import List, Dict, Any, Optional, Tuple, TypedDict, Union
import os
import time
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from rank_bm25 import BM25Okapi
from jet.file.utils import load_file
from jet.models.model_types import EmbedModelType
from jet.models.embeddings.base import load_embed_model
from jet.logger import logger
import re
from tqdm import tqdm
import uuid
from jet.models.tokenizer.base import get_max_token_count
from jet.vectors.document_types import HeaderDocument, HeaderDocumentWithScore
from jet.wordnet.spellcheck import correct_typos
from jet.wordnet.stopwords import StopWords
from jet.wordnet.words import get_words
from llama_index.core.schema import MetadataMode

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


class Match(TypedDict):
    word: str
    start_idx: int
    end_idx: int
    line: str


class SearchResult(TypedDict):
    id: str
    doc_index: int
    rank: int
    score: float
    combined_score: float
    embedding_score: float
    headers: List[str]
    text: str
    highlighted_text: str
    matches: List[Match]


def process_documents(
    documents: Union[List[HeaderDocument], List[Dict[str, Any]], List[str]],
    ids: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    logger.info("Processing %d input objects",
                len(documents) if documents else 0)
    if documents is None:
        logger.error("Received None as documents input")
        raise ValueError("Documents input cannot be None")
    if not isinstance(documents, list):
        logger.error("Input must be a list, got %s", type(documents))
        raise ValueError("Input must be a list")
    result = []
    processed_docs: List[HeaderDocument] = []
    seen_texts = set()
    seen_ids = set()
    if documents and isinstance(documents[0], str):
        logger.info(
            "Converting %d strings to HeaderDocument objects", len(documents))
        for idx, text in enumerate(tqdm(documents, desc="Converting strings")):
            if text in seen_texts:
                continue
            try:
                doc_id = ids[idx] if ids and idx < len(
                    ids) else f"doc_{uuid.uuid4()}"
                if doc_id in seen_ids:
                    doc_id = f"doc_{uuid.uuid4()}"
                header_doc = HeaderDocument(
                    id=doc_id,
                    text=text,
                    metadata={"original_index": idx,
                              "doc_index": idx}  # Assign doc_index
                )
                processed_docs.append(header_doc)
                seen_texts.add(text)
                seen_ids.add(doc_id)
            except Exception as e:
                logger.error(
                    "Failed to convert string to HeaderDocument at index %d: %s", idx, str(e))
                raise ValueError(
                    f"Failed to convert string to HeaderDocument at index {idx}: {str(e)}")
    elif documents and isinstance(documents[0], dict):
        logger.info(
            "Converting %d dictionaries to HeaderDocument objects", len(documents))
        for idx, doc_dict in enumerate(tqdm(documents, desc="Converting dictionaries")):
            try:
                header_doc = HeaderDocument(**doc_dict)
                if header_doc.text in seen_texts:
                    continue
                doc_id = header_doc.id
                if doc_id in seen_ids:
                    doc_id = f"doc_{uuid.uuid4()}"
                    header_doc.id = doc_id
                header_doc.metadata["original_index"] = idx
                header_doc.metadata["doc_index"] = doc_dict.get(
                    "doc_index", idx)  # Preserve doc_index
                processed_docs.append(header_doc)
                seen_texts.add(header_doc.text)
                seen_ids.add(doc_id)
            except Exception as e:
                logger.error(
                    "Failed to convert dictionary to HeaderDocument at index %d: %s", idx, str(e))
                raise ValueError(
                    f"Failed to convert dictionary to HeaderDocument at index {idx}: {str(e)}")
    else:
        logger.info(
            "Processing %d HeaderDocument objects directly", len(documents))
        for idx, doc in enumerate(tqdm(documents, desc="Processing HeaderDocuments")):
            if doc.text in seen_texts:
                continue
            doc_id = doc.id
            if doc_id in seen_ids:
                doc_id = f"doc_{uuid.uuid4()}"
                doc.id = doc_id
            doc.metadata["original_index"] = idx
            doc.metadata["doc_index"] = doc.metadata.get(
                "doc_index", idx)  # Preserve doc_index
            processed_docs.append(doc)
            seen_texts.add(doc.text)
            seen_ids.add(doc_id)
    if ids is not None and len(ids) != len(documents):
        logger.error("Provided IDs length (%d) does not match documents length (%d)", len(
            ids), len(documents))
        raise ValueError("IDs length must match documents length")
    for idx, doc in enumerate(tqdm(processed_docs, desc="Finalizing documents")):
        doc_id = doc.id
        text = doc.text or ""
        metadata = doc.metadata
        if isinstance(metadata, dict) and "header_level" in metadata:
            if metadata.get("header_level") != 1:
                text = "\n".join([
                    f"{metadata["parent_header"] or ""}{"\n" if metadata["parent_header"] else ""}{metadata["header"]}",
                    metadata.get("content", "")
                ]).strip()
                if not text:
                    text = doc.metadata.get("content", "")
        else:
            logger.warning(
                "Document %s lacks metadata, using raw text: %s", doc_id, text[:50])
        result.append({"text": text, "id": doc_id,
                      "index": metadata.get("doc_index", idx)})
    logger.info("Processed %d unique documents", len(result))
    return result


def split_document(doc_text: str, doc_id: str, doc_index: int, chunk_size: int = 800, overlap: int = 200) -> List[Dict[str, Any]]:
    """Split document into semantic chunks using sentence boundaries, preserving newlines between headers and content."""
    logger.debug("Starting split_document for doc_id=%s, doc_index=%d, text_length=%d",
                 doc_id, doc_index, len(doc_text))
    chunks = []
    current_headers = []
    current_chunk = []
    current_content = []
    current_len = 0
    lines = doc_text.strip().split("\n")
    logger.debug("Split into %d lines: %s", len(
        lines), lines[:3] if lines else [])

    for line in lines:
        line = line.strip()
        logger.debug("Processing line: '%s'", line)
        if not line:
            logger.debug("Skipping empty line")
            continue
        if re.match(r'^#{1,6}\s+', line):  # Detect header
            logger.debug("Detected header: '%s'", line)
            if current_content or current_chunk:  # Create chunk if there's content or a previous header
                content_text = " ".join(
                    current_content) if current_content else ""
                chunk_text = "\n".join(current_chunk + [content_text]).strip()
                logger.debug(
                    "Creating chunk with text: '%s', headers: %s", chunk_text, current_headers)
                if chunk_text:
                    chunks.append({
                        "text": chunk_text,
                        "headers": current_headers.copy(),
                        "doc_id": doc_id,
                        "doc_index": doc_index
                    })
                current_content = current_content[-int(
                    overlap / 2):] if overlap else []
                current_len = sum(len(get_words(s)) for s in current_content)
                logger.debug(
                    "After header, reset current_content: %s, current_len: %d", current_content, current_len)
            current_headers = [line]
            current_chunk = [line]
            current_content = []
            current_len += len(get_words(line))
        else:
            sentences = re.split(r'(?<=[.!?])\s+', line.strip())
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    logger.debug("Skipping empty sentence")
                    continue
                sentence_len = len(get_words(sentence))
                logger.debug("Processing sentence: '%s', length: %d",
                             sentence, sentence_len)
                # Handle large sentences by splitting into smaller chunks
                if sentence_len > chunk_size:
                    words = get_words(sentence)
                    header_word_count = len(
                        get_words(current_chunk[0])) if current_chunk else 0
                    available_chunk_size = chunk_size - header_word_count
                    i = 0
                    while i < len(words):
                        if i == 0 and current_chunk:  # First chunk includes header
                            chunk_words = words[i:i + available_chunk_size]
                            content_text = " ".join(chunk_words)
                            chunk_text = "\n".join(
                                current_chunk + [content_text]).strip()
                        else:  # Subsequent chunks
                            chunk_words = words[i:i + chunk_size]
                            content_text = " ".join(chunk_words)
                            chunk_text = content_text
                        if chunk_text:
                            chunks.append({
                                "text": chunk_text,
                                "headers": current_headers.copy(),
                                "doc_id": doc_id,
                                "doc_index": doc_index
                            })
                            logger.debug(
                                "Large sentence split, new chunk: '%s', headers: %s", chunk_text, current_headers)
                        current_content = chunk_words[-int(
                            overlap / 2):] if overlap else []
                        current_len = sum(len(get_words(w))
                                          for w in current_content)
                        current_chunk = current_chunk if overlap and i == 0 else []
                        i += len(chunk_words)
                        logger.debug(
                            "After large sentence chunk, current_content: %s, current_len: %d", current_content, current_len)
                elif current_len + sentence_len > chunk_size and (current_chunk or current_content):
                    content_text = " ".join(
                        current_content) if current_content else ""
                    chunk_text = "\n".join(
                        current_chunk + [content_text]).strip()
                    logger.debug(
                        "Chunk size exceeded, creating chunk with text: '%s'", chunk_text)
                    if chunk_text:
                        chunks.append({
                            "text": chunk_text,
                            "headers": current_headers.copy(),
                            "doc_id": doc_id,
                            "doc_index": doc_index
                        })
                    current_content = [
                        sentence] if not overlap else current_content[-int(overlap / 2):] + [sentence]
                    current_len = sentence_len + \
                        sum(len(get_words(s)) for s in current_content[:-1])
                    current_chunk = current_chunk if overlap else []
                    logger.debug(
                        "After chunk creation, current_content: %s, current_len: %d", current_content, current_len)
                else:
                    current_content.append(sentence)
                    current_len += sentence_len
                    logger.debug(
                        "Added sentence, current_content: %s, current_len: %d", current_content, current_len)

    # Final chunk: Include even if it's just a header or small content
    content_text = " ".join(current_content) if current_content else ""
    chunk_text = "\n".join(current_chunk + [content_text]).strip()
    logger.debug("Final chunk processing, chunk_text: '%s'", chunk_text)
    if chunk_text:  # Always include non-empty final chunk
        chunks.append({
            "text": chunk_text,
            "headers": current_headers.copy(),
            "doc_id": doc_id,
            "doc_index": doc_index
        })
        logger.debug("Appended final chunk: %s", chunks[-1])

    logger.info(
        "Completed splitting for doc_id=%s, created %d chunks", doc_id, len(chunks))
    return chunks


def filter_by_headers(chunks: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """Filter chunks based on header relevance."""
    start_time = time.time()
    logger.info("Filtering chunks by headers for query: %s", query)
    query_terms = set(get_words(query.lower()))
    filtered = []
    for chunk in tqdm(chunks, desc="Filtering chunks"):
        headers = [h.lower() for h in chunk["headers"]]
        if any(any(term in h for term in query_terms) for h in headers):
            filtered.append(chunk)
    duration = time.time() - start_time
    logger.info("Filtering completed in %.3f seconds, reduced %d to %d chunks",
                duration, len(chunks), len(filtered))
    return filtered if filtered else []


def embed_chunk(chunk: str, embedder: SentenceTransformer) -> np.ndarray:
    """Embed a single chunk using SentenceTransformer."""
    embedding = embedder.encode(chunk, convert_to_numpy=True)
    return np.ascontiguousarray(embedding.astype(np.float32))


def embed_chunks_parallel(chunk_texts: List[str], embedder: SentenceTransformer) -> np.ndarray:
    """Embed chunks in batches to optimize performance."""
    start_time = time.time()
    logger.info("Embedding %d chunks in batches", len(chunk_texts))
    if not chunk_texts:
        logger.info("No chunks to embed, returning empty array")
        return np.zeros((0, embedder.get_sentence_embedding_dimension()), dtype=np.float32)
    batch_size = 32
    embeddings = []
    for i in tqdm(range(0, len(chunk_texts), batch_size), desc="Embedding chunks"):
        batch = chunk_texts[i:i + batch_size]
        try:
            batch_embeddings = embedder.encode(
                batch, convert_to_numpy=True, batch_size=batch_size)
            batch_embeddings = np.ascontiguousarray(
                batch_embeddings.astype(np.float32))
            embeddings.extend(batch_embeddings)
        except Exception as e:
            logger.error("Error embedding batch: %s", e)
            for _ in batch:
                embeddings.append(
                    np.zeros(embedder.get_sentence_embedding_dimension(), dtype=np.float32))
    duration = time.time() - start_time
    logger.info("Embedding completed in %.3f seconds", duration)
    result = np.vstack(embeddings)
    return result


def get_original_document(
    doc_id: str,
    doc_index: int,
    documents: Union[List[HeaderDocument], List[Dict[str, Any]], List[str]],
    ids: Optional[List[str]] = None
) -> Optional[HeaderDocument]:
    """Retrieve original HeaderDocument by ID or index."""
    logger.info("Retrieving original document for ID %s, index %d",
                doc_id, doc_index)
    if documents and isinstance(documents[0], str):
        logger.info(
            "Converting %d strings to HeaderDocument objects for retrieval", len(documents))
        for idx, text in enumerate(documents):
            try:
                expected_id = ids[idx] if ids and idx < len(
                    ids) else f"doc_{idx}"
                if doc_id == expected_id or (ids is None and doc_index == idx):
                    header_doc = HeaderDocument(
                        id=expected_id,
                        text=text,
                        metadata={"original_index": idx}
                    )
                    return header_doc
            except Exception as e:
                logger.error("Failed to convert string to HeaderDocument for ID %s at index %d: %s",
                             doc_id, idx, str(e))
                continue
    elif documents and isinstance(documents[0], dict):
        logger.info(
            "Converting %d dictionaries to HeaderDocument objects for retrieval", len(documents))
        for idx, doc_dict in enumerate(documents):
            try:
                header_doc = HeaderDocument(**doc_dict)
                if header_doc.id == doc_id or (ids is None and header_doc.metadata.get("original_index") == doc_index):
                    return header_doc
            except Exception as e:
                logger.error(
                    "Failed to convert dictionary to HeaderDocument for ID %s: %s", doc_id, str(e))
                continue
    else:
        logger.info("Using %d HeaderDocument objects directly for retrieval", len(
            documents) if documents else 0)
        for doc in documents:
            if doc.id == doc_id or (ids is None and doc.metadata.get("original_index") == doc_index):
                return doc
    logger.warning("No HeaderDocument found for ID %s, index %d",
                   doc_id, doc_index)
    return None


def compute_header_similarities(
    chunks: List[Dict[str, Any]],
    query: str,
    embedder: SentenceTransformer
) -> List[float]:
    """Compute cosine similarities between query and concatenated headers for each chunk."""
    logger.info("Computing header similarities for %d chunks", len(chunks))
    header_texts = []
    for chunk in chunks:
        headers = chunk.get("headers", [])
        # Extract parent_header and header from metadata if available
        original_doc = chunk.get("original_doc")
        if original_doc and isinstance(original_doc.metadata, dict):
            parent_header = original_doc.metadata.get("parent_header", "")
            header = original_doc.metadata.get("header", "")
            header_text = f"{parent_header}\n{header}" if parent_header and header else header or parent_header
        else:
            # Fallback to headers list
            header_text = "\n".join(h for h in headers if h) if headers else ""
        header_texts.append(header_text or "")

    if not any(header_texts):
        logger.info("No valid headers found, returning zero similarities")
        return [0.0] * len(chunks)

    # Embed headers in batches
    header_embeddings = embed_chunks_parallel(header_texts, embedder)
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    query_embedding = np.ascontiguousarray(query_embedding.astype(np.float32))
    faiss.normalize_L2(header_embeddings)
    faiss.normalize_L2(query_embedding)

    # Compute cosine similarities
    similarities = np.dot(header_embeddings, query_embedding.T).flatten()
    # Normalize to [0, 1]
    similarities = [(score + 1) / 2 for score in similarities]
    return similarities


def compute_combined_scores(
    embed_scores: List[float],
    bm25_scores: List[float],
    header_scores: List[float],  # Added header_scores
    bm25_weight: float,
    indices: np.ndarray,
    with_bm25: bool
) -> List[float]:
    """Compute combined scores by averaging embedding and header scores, with optional BM25 contribution."""
    combined_scores = []
    for j, i in enumerate(indices[0]):
        bm25_score = bm25_scores[i] if with_bm25 else 0.0
        embed_score = embed_scores[j]
        header_score = header_scores[i]
        # Average content and header scores
        content_header_score = (embed_score + header_score) / 2
        weight = bm25_weight if with_bm25 else 0.0
        combined = weight * bm25_score + (1 - weight) * content_header_score
        combined = max(0.0, min(1.0, combined))
        combined_scores.append(combined)
    return combined_scores


def highlight_text(text: str, query: str) -> Tuple[str, List[Match]]:
    """Highlight non-stopword query terms in text and collect match details."""
    if not query.strip():
        return "", []
    query_terms = get_words(query.lower())
    unique_terms = {
        term.strip(string.punctuation)
        for term in query_terms
        if term not in StopWords.english_stop_words
    }
    if not unique_terms:
        return "", []
    matches: List[Match] = []
    pattern = re.compile(
        r'\b(' + '|'.join(map(re.escape, unique_terms)) + r')\b', re.IGNORECASE
    )

    def highlight_match(match: re.Match) -> str:
        word = match.group(0)
        lower_word = word.lower().strip(string.punctuation)
        if lower_word in unique_terms:
            start_idx = match.start()
            end_idx = match.end()
            line_start = text.rfind('\n', 0, start_idx) + 1
            line_end = text.find('\n', end_idx)
            if line_end == -1:
                line_end = len(text)
            line = text[line_start:line_end].strip()
            matches.append({
                "word": word,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "line": line
            })
            return f"<mark>{word}</mark>"
        return word
    highlighted = pattern.sub(highlight_match, text)
    return highlighted, matches


def search_docs(
    query: str,
    documents: Union[List[HeaderDocument], List[Dict[str, Any]], List[str]],
    instruction: Optional[str] = None,
    ids: Optional[List[str]] = None,
    model: EmbedModelType = "static-retrieval-mrl-en-v1",
    chunk_size: int = 800,
    overlap: int = 200,
    top_k: Optional[int] = 20,
    threshold: float = 0.0,
    filter_by_headers_enabled: bool = False,
    bm25_weight: float = 0.3,
    with_bm25: bool = False,
    split_document_enabled: bool = True  # New parameter to toggle document splitting
) -> List[HeaderDocumentWithScore]:
    """Search documents using embedding-based retrieval, averaging header and content similarities.

    Args:
        query: The search query string.
        documents: List of documents, HeaderDocument objects, dictionaries, or strings.
        instruction: Optional instruction to prepend to the query.
        ids: Optional list of document IDs.
        model: Embedding model name (default: "static-retrieval-mrl-en-v1").
        chunk_size: Size of document chunks (default: 800).
        overlap: Overlap between chunks (default: 200).
        top_k: Number of top results to retrieve (default: 20).
        threshold: Minimum score threshold for results (default: 0.0).
        filter_by_headers_enabled: Whether to filter chunks by headers (default: False).
        bm25_weight: Weight for BM25 score in combined score (default: 0.3).
        with_bm25: Whether to include BM25 scores (default: False).
        split_document_enabled: Whether to split documents into chunks (default: True).

    Returns:
        List of HeaderDocumentWithScore.
    """
    total_start_time = time.time()
    if instruction and isinstance(instruction, str) and instruction.strip():
        logger.info("Using instruction: %s", instruction)
        query_with_instruction = f"{instruction} {query}".strip()
    else:
        logger.warning("Invalid or empty instruction, using query alone")
        query_with_instruction = query

    docs = process_documents(documents, ids)
    start_time = time.time()

    if split_document_enabled:
        chunks = []
        for doc in tqdm(docs, desc="Splitting documents"):
            chunks.extend(split_document(
                doc["text"], doc["id"], doc["index"], chunk_size, overlap))
        logger.info("Document splitting completed in %.3f seconds, created %d chunks",
                    time.time() - start_time, len(chunks))
    else:
        # Use original documents directly without splitting
        chunks = [
            {
                "text": doc["text"],
                "headers": [],  # No headers when splitting is disabled
                "doc_id": doc["id"],
                "doc_index": doc["index"]
            }
            for doc in docs
        ]
        logger.info("Document splitting disabled, using %d original documents as chunks",
                    len(chunks))

    filtered_chunks = filter_by_headers(
        chunks, query) if filter_by_headers_enabled else chunks
    chunk_texts = [chunk["text"] for chunk in filtered_chunks]

    logger.info("Initializing SentenceTransformer")
    max_token_length = get_max_token_count(model, chunk_texts)
    embedder = load_embed_model(model, max_token_length)

    # Embed chunks and compute header similarities
    chunk_embeddings = embed_chunks_parallel(chunk_texts, embedder)
    header_scores = compute_header_similarities(
        filtered_chunks, query_with_instruction, embedder)

    # Compute BM25 scores if enabled
    bm25_scores = []
    if with_bm25:
        tokenized_chunks = [get_words(chunk["text"].lower())
                            for chunk in filtered_chunks]
        bm25 = BM25Okapi(tokenized_chunks, k1=1.5, b=0.25, epsilon=0.1)
        tokenized_query = get_words(query_with_instruction.lower())
        bm25_scores = bm25.get_scores(tokenized_query).tolist()
        # Normalize BM25 scores to [0, 1]
        max_bm25 = max(bm25_scores) if bm25_scores else 1.0
        bm25_scores = [score / max_bm25 if max_bm25 >
                       0 else 0.0 for score in bm25_scores]

    top_k = min(top_k or len(chunk_texts), len(chunk_texts))
    logger.info("Performing FAISS search with top-k=%d", top_k)
    dim = chunk_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(chunk_embeddings)
    index.add(chunk_embeddings)
    query_embedding = embedder.encode(
        [query_with_instruction], convert_to_numpy=True)
    query_embedding = np.ascontiguousarray(query_embedding.astype(np.float32))
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, top_k)
    embed_scores = [(score + 1) / 2 for score in distances[0].tolist()]

    # Compute combined scores with header similarities
    combined_scores = compute_combined_scores(
        embed_scores, bm25_scores, header_scores, bm25_weight, indices, with_bm25
    )

    initial_docs = [(filtered_chunks[i], embed_scores[j], combined_scores[j])
                    for j, i in enumerate(indices[0])]
    sorted_indices = np.argsort([doc[2] for doc in initial_docs])[::-1][:top_k]
    sorted_docs = [(initial_docs[i][0], initial_docs[i][1], initial_docs[i][2])
                   for i in sorted_indices]

    results: List[HeaderDocumentWithScore] = []
    seen_doc_ids = set()
    for rank, (chunk, embedding_score, combined_score) in enumerate(sorted_docs, 1):
        if combined_score < threshold:
            continue
        doc_id = chunk["doc_id"]
        doc_index = chunk["doc_index"]
        if doc_id in seen_doc_ids:
            continue
        original_doc = get_original_document(doc_id, doc_index, documents, ids)
        if original_doc is None:
            logger.warning(
                "Skipping doc_id %s: original document not found", doc_id)
            continue
        # Attach original_doc to chunk for header similarity computation
        chunk["original_doc"] = original_doc
        highlighted_text, matches = highlight_text(original_doc.text, query)
        result = HeaderDocumentWithScore(
            node=original_doc,
            score=combined_score,
            doc_index=doc_index,
            rank=rank,
            combined_score=combined_score,
            embedding_score=embedding_score,
            headers=chunk["headers"],
            highlighted_text=highlighted_text,
            matches=matches
        )
        results.append(result)
        seen_doc_ids.add(doc_id)

    logger.info("Total unique results: %d", len(results))
    logger.info("Search completed in %.3f seconds",
                time.time() - total_start_time)
    return results
