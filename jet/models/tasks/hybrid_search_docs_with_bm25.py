import os
import time
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from rank_bm25 import BM25Okapi
from jet.file.utils import load_file
from jet.logger import logger
from typing import List, Dict, Any, Optional, TypedDict, Union
import re
from tqdm import tqdm
import uuid

from jet.vectors.document_types import HeaderDocument

# Set environment variables before importing numpy/pytorch
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
# Fallback to CPU for unsupported MPS ops
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
    """Process and preprocess HeaderDocument objects, dictionaries, or strings."""
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

    # Handle List[str]
    if documents and isinstance(documents[0], str):
        logger.info(
            "Converting %d strings to HeaderDocument objects", len(documents))
        for idx, text in enumerate(tqdm(documents, desc="Converting strings")):
            try:
                header_doc = HeaderDocument(
                    id=ids[idx] if ids and idx < len(
                        ids) else str(uuid.uuid4()),
                    text=text,
                    metadata={}
                )
                processed_docs.append(header_doc)
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
                processed_docs.append(header_doc)
            except Exception as e:
                logger.error(
                    "Failed to convert dictionary to HeaderDocument at index %d: %s", idx, str(e))
                raise ValueError(
                    f"Failed to convert dictionary to HeaderDocument at index {idx}: {str(e)}")
    # Handle List[HeaderDocument]
    else:
        logger.info(
            "Processing %d HeaderDocument objects directly", len(documents))
        processed_docs = documents  # Type: List[HeaderDocument]

    if ids is not None and len(ids) != len(processed_docs):
        logger.error("Provided IDs length (%d) does not match documents length (%d)", len(
            ids), len(processed_docs))
        raise ValueError("IDs length must match documents length")

    for idx, doc in enumerate(tqdm(processed_docs, desc="Processing documents")):
        # Use existing doc.id if available, otherwise use ids[idx] or generate UUID
        doc_id = doc.id if doc.id else (
            ids[idx] if ids and idx < len(ids) else str(uuid.uuid4()))
        logger.debug("Assigned ID %s to document at index %d", doc_id, idx)
        metadata = doc.metadata
        if isinstance(metadata, dict) and "header_level" in metadata:
            if metadata.get("header_level") != 1:
                text = "\n".join([
                    metadata.get("parent_header", ""),
                    metadata.get("header", ""),
                    metadata.get("content", "")
                ]).strip()
                result.append({"text": text, "id": doc_id, "index": idx})
            else:
                text = doc.get_recursive_text().strip()
                result.append({"text": text, "id": doc_id, "index": idx})
        else:
            text = doc.text or ""
            logger.warning(
                "Document %s lacks metadata, using text: %s", doc_id, text[:50])
            result.append({"text": text, "id": doc_id, "index": idx})
    return result


def split_document(doc_text: str, doc_id: str, doc_index: int, chunk_size: int = 800, overlap: int = 200) -> List[Dict[str, Any]]:
    """Split document into semantic chunks using sentence boundaries."""
    logger.info("Splitting document ID %s (index %d) into chunks with chunk_size=%d, overlap=%d",
                doc_id, doc_index, chunk_size, overlap)
    chunks = []
    current_headers = []
    current_chunk = []
    current_len = 0

    lines = doc_text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith(("# ", "## ")):
            logger.debug("Processing header: %s", line)
            if current_chunk:
                chunk_text = " ".join(current_chunk).strip()
                logger.debug("Creating chunk with text: %s, headers: %s, len: %d",
                             chunk_text, current_headers, current_len)
                chunks.append({
                    "text": chunk_text,
                    "headers": current_headers.copy(),
                    "doc_id": doc_id,
                    "doc_index": doc_index
                })
                current_chunk = [] if not overlap else current_chunk[-1:]
                current_len = sum(len(s.split()) for s in current_chunk)
            current_headers = [line]
            current_chunk.append(line)
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
                if current_chunk and (current_len + sentence_len >= chunk_size or len(current_chunk) > 1):
                    chunk_text = " ".join(current_chunk).strip()
                    logger.debug("Creating chunk with text: %s, headers: %s, len: %d",
                                 chunk_text, current_headers, current_len)
                    chunks.append({
                        "text": chunk_text,
                        "headers": current_headers.copy(),
                        "doc_id": doc_id,
                        "doc_index": doc_index
                    })
                    current_chunk = [sentence] if not overlap else [
                        current_chunk[-1], sentence] if current_chunk else [sentence]
                    current_len = sentence_len + \
                        (sum(len(s.split())
                         for s in current_chunk[:-1]) if overlap else 0)
                else:
                    current_chunk.append(sentence)
                    current_len += sentence_len

    if current_chunk:
        chunk_text = " ".join(current_chunk).strip()
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
        if any(any(term in h for term in query_terms) for h in headers) or not headers:
            filtered.append(chunk)
    duration = time.time() - start_time
    logger.info("Filtering completed in %.3f seconds, reduced %d to %d chunks",
                duration, len(chunks), len(filtered))
    return filtered if filtered else chunks


def embed_chunk(chunk: str, embedder: SentenceTransformer) -> np.ndarray:
    """Embed a single chunk using SentenceTransformer."""
    return embedder.encode(chunk, convert_to_numpy=True)


def embed_chunks_parallel(chunk_texts: List[str], embedder: SentenceTransformer) -> np.ndarray:
    """Embed chunks in batches to optimize performance."""
    start_time = time.time()
    logger.info("Embedding %d chunks in batches", len(chunk_texts))
    logger.debug("Embedding device: %s", embedder.device)
    batch_size = 32  # Adjustable based on memory
    embeddings = []
    for i in tqdm(range(0, len(chunk_texts), batch_size), desc="Embedding chunks"):
        batch = chunk_texts[i:i + batch_size]
        try:
            batch_embeddings = embedder.encode(
                batch, convert_to_numpy=True, batch_size=batch_size)
            embeddings.extend(batch_embeddings)
        except Exception as e:
            logger.error("Error embedding batch: %s", e)
            for _ in batch:
                embeddings.append(
                    np.zeros(embedder.get_sentence_embedding_dimension()))
    duration = time.time() - start_time
    logger.info("Embedding completed in %.3f seconds", duration)
    return np.vstack(embeddings)


def get_bm25_scores(chunk_texts: List[str], query: str) -> List[float]:
    """Calculate BM25 scores for chunks."""
    tokenized_chunks = [text.lower().split() for text in chunk_texts]
    bm25 = BM25Okapi(tokenized_chunks)
    tokenized_query = query.lower().split()
    return bm25.get_scores(tokenized_query).tolist()


def get_original_document(
    doc_id: str,
    documents: Union[List[HeaderDocument], List[Dict[str, Any]], List[str]]
) -> Optional[HeaderDocument]:
    """Retrieve original HeaderDocument by ID."""
    logger.info("Retrieving original HeaderDocument for ID %s", doc_id)
    processed_docs: List[HeaderDocument] = []

    if documents and isinstance(documents[0], str):
        logger.info(
            "Converting %d strings to HeaderDocument objects for retrieval", len(documents))
        for idx, text in enumerate(documents):
            try:
                header_doc = HeaderDocument(
                    id=doc_id if doc_id == (ids[idx] if ids and idx < len(
                        ids) else str(uuid.uuid4())) else str(uuid.uuid4()),
                    text=text,
                    metadata={}
                )
                processed_docs.append(header_doc)
            except Exception as e:
                logger.error(
                    "Failed to convert string to HeaderDocument for ID %s: %s", doc_id, str(e))
                continue
    elif documents and isinstance(documents[0], dict):
        logger.info(
            "Converting %d dictionaries to HeaderDocument objects for retrieval", len(documents))
        for doc_dict in documents:
            try:
                header_doc = HeaderDocument(**doc_dict)
                processed_docs.append(header_doc)
            except Exception as e:
                logger.error(
                    "Failed to convert dictionary to HeaderDocument for ID %s: %s", doc_id, str(e))
                continue
    else:
        logger.info("Using %d HeaderDocument objects directly for retrieval", len(
            documents) if documents else 0)
        processed_docs = documents  # Type: List[HeaderDocument]

    logger.debug("Processed docs IDs: %s", [
                 doc.id for doc in processed_docs if doc.id])
    for doc in processed_docs:
        if doc.id == doc_id:
            logger.debug("Found HeaderDocument for ID %s", doc_id)
            return doc
    logger.warning("No HeaderDocument found for ID %s", doc_id)
    return None


def search_docs(
    query: str,
    documents: Union[List[HeaderDocument], List[Dict[str, Any]], List[str]],
    instruction: Optional[str] = None,
    ids: Optional[List[str]] = None,
    model: str = "static-retrieval-mrl-en-v1",
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    chunk_size: int = 800,
    overlap: int = 200,
    top_k: int = 20,
    rerank_top_k: int = 5,
    batch_size: int = 8,
    bm25_weight: float = 0.5
) -> List[SearchResult]:
    """Search documents using hybrid retrieval with BM25 and embeddings, incorporating an instruction."""
    total_start_time = time.time()
    logger.debug("Input document IDs: %s", [
        doc.id if isinstance(doc, HeaderDocument) else doc.get(
            "id") if isinstance(doc, dict) else str(uuid.uuid4())
        for doc in documents if doc
    ])
    logger.info("Using instruction: %s", instruction)

    # Validate instruction
    if not isinstance(instruction, str) or not instruction.strip():
        logger.warning(
            "Invalid or empty instruction provided, proceeding without instruction")
        instruction = ""

    docs = process_documents(documents, ids)

    # Splitting documents
    start_time = time.time()
    chunks = []
    for doc in tqdm(docs, desc="Splitting documents"):
        chunks.extend(split_document(
            doc["text"], doc["id"], doc["index"], chunk_size, overlap))
    split_duration = time.time() - start_time
    logger.info("Document splitting completed in %.3f seconds, created %d chunks",
                split_duration, len(chunks))

    # Filtering chunks
    filtered_chunks = filter_by_headers(chunks, query)
    chunk_texts = [chunk["text"] for chunk in filtered_chunks]

    # Embedding chunks
    logger.info("Initializing SentenceTransformer and CrossEncoder models")
    embedder = SentenceTransformer(
        model, device="cpu", backend="onnx")  # Use ONNX on CPU
    cross_encoder = CrossEncoder(rerank_model)
    chunk_embeddings = embed_chunks_parallel(chunk_texts, embedder)

    # FAISS search
    start_time = time.time()
    dim = chunk_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Use inner product for cosine similarity
    faiss.normalize_L2(chunk_embeddings)  # Normalize for cosine similarity
    index.add(chunk_embeddings)
    top_k = min(top_k, len(chunk_texts))
    logger.info("Performing FAISS search with top-k=%d", top_k)

    # Combine instruction with query for embedding
    query_with_instruction = f"{instruction} {query}".strip(
    ) if instruction else query
    logger.debug("Query with instruction: %s", query_with_instruction)
    query_embedding = embedder.encode(
        [query_with_instruction], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)  # Normalize query
    logger.debug("Query embedding norm: %.4f", np.linalg.norm(query_embedding))
    distances, indices = index.search(query_embedding, top_k)
    embed_scores = distances[0].tolist()  # Cosine similarity [0, 1]
    logger.debug("FAISS distances: %s", distances[0].tolist())
    logger.debug("Embed scores: %s", embed_scores)
    faiss_duration = time.time() - start_time
    logger.info("FAISS search completed in %.3f seconds", faiss_duration)

    # Combine scores
    bm25_scores = get_bm25_scores(chunk_texts, query)
    combined_scores = [
        bm25_weight * bm25_scores[i] + (1 - bm25_weight) * embed_scores[j]
        for j, i in enumerate(indices[0])
    ]
    initial_docs = [
        (filtered_chunks[i], combined_scores[j], embed_scores[j])
        for j, i in enumerate(indices[0])
    ]

    # Reranking
    start_time = time.time()
    logger.info("Reranking %d documents with cross-encoder", len(initial_docs))
    # Use instruction in reranking
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
    reranked_indices = np.argsort(scores)[::-1][:rerank_top_k]
    reranked_docs = [
        (initial_docs[i][0], initial_docs[i][1], scores[i], initial_docs[i][2])
        for i in reranked_indices
    ]
    rerank_duration = time.time() - start_time
    logger.info("Reranking completed in %.3f seconds", rerank_duration)

    # Build results
    results: List[SearchResult] = []
    seen_doc_ids = set()
    for i, (chunk, combined_score, rerank_score, embedding_score) in enumerate(reranked_docs):
        doc_id = chunk["doc_id"]
        logger.debug("Processing chunk with doc_id %s", doc_id)
        if doc_id not in seen_doc_ids:
            original_doc = get_original_document(doc_id, documents)
            if original_doc is None:
                logger.warning(
                    "Skipping result for ID %s: original document not found", doc_id)
                continue
            result: SearchResult = {
                "id": doc_id,
                "doc_index": chunk["doc_index"],
                "rank": i + 1,
                "score": rerank_score,
                "combined_score": combined_score,
                "embedding_score": embedding_score,
                "headers": chunk["headers"],
                "text": original_doc.text,
                "document": original_doc
            }
            results.append(result)
            seen_doc_ids.add(doc_id)

    total_duration = time.time() - total_start_time
    logger.info("Total search completed in %.3f seconds", total_duration)
    return results
