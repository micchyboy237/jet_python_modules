from typing import List, Dict, Any, Optional, TypedDict, Union
import os
import time
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from rank_bm25 import BM25Okapi
from jet.file.utils import load_file
from jet.logger import logger
import re
from tqdm import tqdm
import uuid
from jet.vectors.document_types import HeaderDocument

# Set environment variables for Mac M1 compatibility
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


class SearchResult(TypedDict):
    id: str
    doc_index: int
    rank: int
    score: float
    combined_score: float
    embedding_score: float
    headers: List[str]
    text: str
    document: HeaderDocument


def process_documents(
    documents: Union[List[HeaderDocument], List[Dict[str, Any]], List[str]],
    ids: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Process and preprocess HeaderDocument objects, dictionaries, or strings.
    Ensures unique document IDs and removes duplicate documents based on text content.
    """
    logger.info("Processing %d input objects",
                len(documents) if documents else 0)
    logger.debug("Input documents type: %s, value: %s",
                 type(documents), documents)

    if documents is None:
        logger.error("Received None as documents input")
        raise ValueError("Documents input cannot be None")

    if not isinstance(documents, list):
        logger.error("Input must be a list, got %s", type(documents))
        raise ValueError("Input must be a list")

    result = []
    processed_docs: List[HeaderDocument] = []
    seen_texts = set()  # Track unique document texts
    seen_ids = set()    # Track unique document IDs

    # Handle List[str]
    if documents and isinstance(documents[0], str):
        logger.info(
            "Converting %d strings to HeaderDocument objects", len(documents))
        for idx, text in enumerate(tqdm(documents, desc="Converting strings")):
            if text in seen_texts:
                logger.debug(
                    "Skipping duplicate text at index %d: %s", idx, text[:50])
                continue
            try:
                doc_id = ids[idx] if ids and idx < len(
                    ids) else f"doc_{uuid.uuid4()}"
                if doc_id in seen_ids:
                    logger.debug(
                        "Duplicate ID %s detected, generating new UUID", doc_id)
                    doc_id = f"doc_{uuid.uuid4()}"
                header_doc = HeaderDocument(
                    id=doc_id,
                    text=text,
                    metadata={"original_index": idx}
                )
                processed_docs.append(header_doc)
                seen_texts.add(text)
                seen_ids.add(doc_id)
                logger.debug(
                    "Processed document ID %s at index %d", doc_id, idx)
            except Exception as e:
                logger.error(
                    "Failed to convert string to HeaderDocument at index %d: %s", idx, str(e))
                raise ValueError(
                    f"Failed to convert string to HeaderDocument at index {idx}: {str(e)}")
    # Handle List[Dict[str, Any]]
    elif documents and isinstance(documents[0], dict):
        logger.info(
            "Converting %d dictionaries to HeaderDocument objects", len(documents))
        for idx, doc_dict in enumerate(tqdm(documents, desc="Converting dictionaries")):
            try:
                header_doc = HeaderDocument(**doc_dict)
                if header_doc.text in seen_texts:
                    logger.debug(
                        "Skipping duplicate text at index %d: %s", idx, header_doc.text[:50])
                    continue
                doc_id = header_doc.id
                if doc_id in seen_ids:
                    logger.debug(
                        "Duplicate ID %s detected, generating new UUID", doc_id)
                    doc_id = f"doc_{uuid.uuid4()}"
                    header_doc.id = doc_id
                header_doc.metadata["original_index"] = idx
                processed_docs.append(header_doc)
                seen_texts.add(header_doc.text)
                seen_ids.add(doc_id)
                logger.debug(
                    "Processed document ID %s at index %d", doc_id, idx)
            except Exception as e:
                logger.error(
                    "Failed to convert dictionary to HeaderDocument at index %d: %s", idx, str(e))
                raise ValueError(
                    f"Failed to convert dictionary to HeaderDocument at index {idx}: {str(e)}")
    # Handle List[HeaderDocument]
    else:
        logger.info(
            "Processing %d HeaderDocument objects directly", len(documents))
        for idx, doc in enumerate(tqdm(documents, desc="Processing HeaderDocuments")):
            if doc.text in seen_texts:
                logger.debug(
                    "Skipping duplicate text at index %d: %s", idx, doc.text[:50])
                continue
            doc_id = doc.id
            if doc_id in seen_ids:
                logger.debug(
                    "Duplicate ID %s detected, generating new UUID", doc_id)
                doc_id = f"doc_{uuid.uuid4()}"
                doc.id = doc_id
            doc.metadata["original_index"] = idx
            processed_docs.append(doc)
            seen_texts.add(doc.text)
            seen_ids.add(doc_id)
            logger.debug("Processed document ID %s at index %d", doc_id, idx)

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
                    metadata.get("parent_header") or "",
                    metadata.get("header", ""),
                    metadata.get("content", "")
                ]).strip()
                if not text:
                    text = doc.text
                logger.debug(
                    "Constructed text for document ID %s with metadata: %s", doc_id, text[:50])
        else:
            logger.warning(
                "Document %s lacks metadata, using raw text: %s", doc_id, text[:50])
        result.append({"text": text, "id": doc_id, "index": idx})
        logger.debug("Finalized document ID %s at index %d", doc_id, idx)

    logger.info("Processed %d unique documents", len(result))
    return result


def split_document(doc_text: str, doc_id: str, doc_index: int, chunk_size: int = 800, overlap: int = 200) -> List[Dict[str, Any]]:
    """Split document into semantic chunks using sentence boundaries, preserving newlines between headers and content."""
    logger.info("Splitting document ID %s (index %d) into chunks with chunk_size=%d, overlap=%d",
                doc_id, doc_index, chunk_size, overlap)
    chunks = []
    current_headers = []
    current_chunk = []
    current_content = []
    current_len = 0

    lines = doc_text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.match(r'^#{1,2}\s+', line):
            logger.debug("Processing header: %s", line)
            if current_chunk or current_content:
                chunk_text = "\n".join(current_chunk + current_content).strip()
                logger.debug("Creating chunk with text: %s, headers: %s, len: %d",
                             chunk_text, current_headers, current_len)
                chunks.append({
                    "text": chunk_text,
                    "headers": current_headers.copy(),
                    "doc_id": doc_id,
                    "doc_index": doc_index
                })
                current_content = [
                ] if not overlap else current_content[-int(overlap / 2):]
                current_chunk = [] if not overlap else current_chunk
                current_len = sum(len(s.split())
                                  for s in current_chunk + current_content)
            current_headers = [line]
            current_chunk = [line]
            current_content = []
            current_len += len(line.split())
        else:
            sentences = re.split(r'(?<=[.!?])\s+', line.strip())
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                sentence_len = len(sentence.split())
                logger.debug("Processing sentence: %s, len: %d, current_len: %d",
                             sentence, sentence_len, current_len)
                if (current_chunk or current_content) and current_len + sentence_len > chunk_size:
                    chunk_text = "\n".join(
                        current_chunk + current_content).strip()
                    logger.debug("Creating chunk with text: %s, headers: %s, len: %d",
                                 chunk_text, current_headers, current_len)
                    chunks.append({
                        "text": chunk_text,
                        "headers": current_headers.copy(),
                        "doc_id": doc_id,
                        "doc_index": doc_index
                    })
                    current_content = [
                        sentence] if not overlap else current_content[-int(overlap / 2):] + [sentence]
                    current_chunk = current_chunk if overlap else []
                    current_len = sentence_len + \
                        sum(len(s.split())
                            for s in current_chunk + current_content[:-1])
                else:
                    current_content.append(sentence)
                    current_len += sentence_len

    if current_chunk or current_content:
        chunk_text = "\n".join(current_chunk + current_content).strip()
        if chunk_text:
            logger.debug("Creating final chunk with text: %s, headers: %s, len: %d",
                         chunk_text, current_headers, current_len)
            chunks.append({
                "text": chunk_text,
                "headers": current_headers.copy(),
                "doc_id": doc_id,
                "doc_index": doc_index
            })

    logger.info("Created %d chunks for document ID %s (index %d)",
                len(chunks), doc_id, doc_index)
    return chunks


def filter_by_headers(chunks: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """Filter chunks based on header relevance."""
    start_time = time.time()
    logger.info("Filtering chunks by headers for query: %s", query)
    query_terms = set(query.lower().split())
    filtered = []
    for chunk in tqdm(chunks, desc="Filtering chunks"):
        headers = [h.lower() for h in chunk["headers"]]
        logger.debug("Checking headers: %s against query terms: %s",
                     headers, query_terms)
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
    logger.debug("Embedding device: %s", embedder.device)
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
            logger.debug("Batch embeddings shape: %s, dtype: %s",
                         batch_embeddings.shape, batch_embeddings.dtype)
            embeddings.extend(batch_embeddings)
        except Exception as e:
            logger.error("Error embedding batch: %s", e)
            for _ in batch:
                embeddings.append(
                    np.zeros(embedder.get_sentence_embedding_dimension(), dtype=np.float32))
    duration = time.time() - start_time
    logger.info("Embedding completed in %.3f seconds", duration)
    result = np.vstack(embeddings)
    logger.debug("Final embeddings shape: %s, dtype: %s",
                 result.shape, result.dtype)
    return result


def get_bm25_scores(chunk_texts: List[str], query: str) -> List[float]:
    """Calculate BM25 scores for chunks, handling edge cases."""
    logger.info("Calculating BM25 scores for %d chunks", len(chunk_texts))
    if not chunk_texts or not query.strip():
        logger.warning("Empty chunk_texts or query, returning zero scores")
        return [0.0] * len(chunk_texts)

    # Tokenize chunks and query, ensuring non-empty tokens
    tokenized_chunks = []
    for i, text in enumerate(chunk_texts):
        tokens = text.lower().split()
        if not tokens:
            logger.warning("Empty tokens for chunk %d, using empty list", i)
            tokenized_chunks.append([])
        else:
            tokenized_chunks.append(tokens)

    tokenized_query = query.lower().split()
    if not tokenized_query:
        logger.warning("Empty query tokens, returning zero scores")
        return [0.0] * len(chunk_texts)

    try:
        bm25 = BM25Okapi(tokenized_chunks)
        scores = bm25.get_scores(tokenized_query).tolist()
        logger.debug("Raw BM25 scores: %s", scores)
        # Ensure non-negative scores
        scores = [max(0.0, score) for score in scores]
        logger.debug("Clamped BM25 scores: %s", scores)
        return scores
    except Exception as e:
        logger.error("Error in BM25 scoring: %s", e)
        return [0.0] * len(chunk_texts)


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
                    logger.debug(
                        "Matched document at index %d with ID %s", idx, expected_id)
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
                    logger.debug(
                        "Found HeaderDocument for ID %s at index %d", doc_id, idx)
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
                logger.debug("Found HeaderDocument for ID %s at index %d",
                             doc_id, doc.metadata.get("original_index"))
                return doc

    logger.warning("No HeaderDocument found for ID %s, index %d",
                   doc_id, doc_index)
    return None


def search_docs(
    query: str,
    documents: Union[List[HeaderDocument], List[Dict[str, Any]], List[str]],
    instruction: Optional[str] = None,
    ids: Optional[List[str]] = None,
    model: str = "static-retrieval-mrl-en-v1",
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
    chunk_size: int = 800,
    overlap: int = 200,
    top_k: Optional[int] = 20,
    rerank_top_k: Optional[int] = 10,
    batch_size: int = 8,
    bm25_weight: float = 0.5
) -> List[SearchResult]:
    """Search documents using hybrid retrieval with BM25 and embeddings, incorporating an instruction.
    Ensures unique results by document ID, selecting next top result for duplicates.
    """
    total_start_time = time.time()
    logger.debug("Input document IDs: %s", [
        doc.id if isinstance(doc, HeaderDocument) else doc.get(
            "id") if isinstance(doc, dict) else str(uuid.uuid4())
        for doc in documents if doc
    ])

    if instruction:
        logger.info("Using instruction: %s", instruction)

    if not isinstance(instruction, str) or not instruction.strip():
        logger.warning(
            "Invalid or empty instruction provided, proceeding without instruction")
        instruction = ""

    docs = process_documents(documents, ids)

    start_time = time.time()
    chunks = []
    for doc in tqdm(docs, desc="Splitting documents"):
        chunks.extend(split_document(
            doc["text"], doc["id"], doc["index"], chunk_size, overlap))
    split_duration = time.time() - start_time
    logger.info("Document splitting completed in %.3f seconds, created %d chunks",
                split_duration, len(chunks))

    filtered_chunks = filter_by_headers(chunks, query)
    chunk_texts = [chunk["text"] for chunk in filtered_chunks]
    logger.debug("Filtered chunks: %s", [
        {"index": i, "doc_id": chunk["doc_id"],
            "text": chunk["text"][:50] + "..."}
        for i, chunk in enumerate(filtered_chunks)
    ])

    logger.info("Initializing SentenceTransformer and CrossEncoder models")
    embedder = SentenceTransformer(model, device="cpu", backend="onnx")
    cross_encoder = CrossEncoder(rerank_model, device="cpu", backend="onnx")
    chunk_embeddings = embed_chunks_parallel(chunk_texts, embedder)

    start_time = time.time()
    dim = chunk_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(chunk_embeddings)
    index.add(chunk_embeddings)

    if not top_k:
        top_k = len(chunk_texts)
    top_k = min(top_k, len(chunk_texts))
    logger.info("Performing FAISS search with top-k=%d", top_k)

    query_with_instruction = f"{instruction} {query}".strip(
    ) if instruction else query
    logger.debug("Query with instruction: %s", query_with_instruction)
    query_embedding = embedder.encode(
        [query_with_instruction], convert_to_numpy=True)
    query_embedding = np.ascontiguousarray(query_embedding.astype(np.float32))
    logger.debug("Query embedding shape: %s, dtype: %s",
                 query_embedding.shape, query_embedding.dtype)
    faiss.normalize_L2(query_embedding)
    logger.debug("Query embedding norm: %.4f", np.linalg.norm(query_embedding))
    distances, indices = index.search(query_embedding, top_k)
    embed_scores = distances[0].tolist()
    logger.debug("FAISS distances: %s", embed_scores)

    # Normalize embedding scores to [0, 1]
    embed_scores = [(score + 1) / 2 for score in embed_scores]
    logger.debug("Normalized embed scores: %s", embed_scores)

    # Calculate BM25 scores
    bm25_scores = get_bm25_scores(chunk_texts, query)
    logger.debug("BM25 scores for chunks: %s", [
        {"chunk_index": i, "score": score} for i, score in enumerate(bm25_scores)
    ])

    # Normalize BM25 scores to [0, 1]
    max_bm25 = max(bm25_scores) if bm25_scores and max(
        bm25_scores) > 0 else 1.0
    bm25_scores = [score / max_bm25 for score in bm25_scores]
    logger.debug("Normalized BM25 scores: %s", [
        {"chunk_index": i, "score": score} for i, score in enumerate(bm25_scores)
    ])

    # Calculate combined scores with detailed logging
    combined_scores = []
    for j, i in enumerate(indices[0]):
        bm25_score = bm25_scores[i]
        embed_score = embed_scores[j]
        combined = bm25_weight * bm25_score + (1 - bm25_weight) * embed_score
        combined = max(0.0, min(1.0, combined))
        combined_scores.append(combined)
        logger.debug(
            "Combined score for chunk %d (FAISS index %d): BM25=%.4f, Embed=%.4f, Combined=%.4f",
            i, j, bm25_score, embed_score, combined
        )

    initial_docs = [
        (filtered_chunks[i], combined_scores[j], embed_scores[j])
        for j, i in enumerate(indices[0])
    ]

    start_time = time.time()
    logger.info("Reranking %d documents with cross-encoder", len(initial_docs))
    pairs = [[query_with_instruction, doc["text"]]
             for doc, _, _ in initial_docs]
    scores = []
    try:
        for i in tqdm(range(0, len(pairs), batch_size), desc="Reranking"):
            batch = pairs[i:i + batch_size]
            batch_scores = cross_encoder.predict(batch)
            scores.extend(batch_scores)
    except Exception as e:
        logger.error("Error in reranking: %s", e)
        scores = [0] * len(pairs)
        logger.info(
            "Added %d zeros as default scores due to reranking error", len(pairs))

    if not rerank_top_k:
        rerank_top_k = len(initial_docs)
    reranked_indices = np.argsort(
        scores)[::-1][:max(rerank_top_k, len(initial_docs))]
    reranked_docs = [
        (initial_docs[i][0], initial_docs[i][1], scores[i], initial_docs[i][2])
        for i in reranked_indices
    ]
    rerank_duration = time.time() - start_time
    logger.info("Reranking completed in %.3f seconds", rerank_duration)

    results: List[SearchResult] = []
    seen_doc_ids = set()
    candidate_index = 0
    while len(results) < rerank_top_k and candidate_index < len(reranked_docs):
        chunk, combined_score, rerank_score, embedding_score = reranked_docs[candidate_index]
        doc_id = chunk["doc_id"]
        doc_index = chunk["doc_index"]
        logger.debug(
            "Processing chunk with doc_id %s, doc_index %d, rank %d, combined_score=%.4f",
            doc_id, doc_index, candidate_index + 1, combined_score
        )

        if doc_id not in seen_doc_ids:
            original_doc = get_original_document(
                doc_id, doc_index, documents, ids)
            if original_doc is None:
                logger.warning(
                    "Skipping result for ID %s: original document not found", doc_id)
                candidate_index += 1
                continue
            result: SearchResult = {
                "id": doc_id,
                "doc_index": doc_index,
                "rank": len(results) + 1,
                "score": rerank_score,
                "combined_score": combined_score,
                "embedding_score": embedding_score,
                "headers": chunk["headers"],
                "text": original_doc.text,
                "document": original_doc
            }
            results.append(result)
            seen_doc_ids.add(doc_id)
            logger.debug(
                "Added unique result for doc_id %s, total unique results: %d", doc_id, len(results))
        else:
            logger.debug(
                "Skipping duplicate doc_id %s at candidate index %d", doc_id, candidate_index)
        candidate_index += 1

    logger.info("Total unique results returned: %d", len(results))
    total_duration = time.time() - total_start_time
    logger.info("Total search completed in %.3f seconds", total_duration)
    return results
