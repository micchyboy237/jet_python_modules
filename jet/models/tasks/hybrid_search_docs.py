import os
import time
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from rank_bm25 import BM25Okapi
from jet.file.utils import load_file
from jet.logger import logger
from typing import List, Dict, Any, Optional
import re
from tqdm import tqdm
import uuid

# Set environment variables before importing numpy/pytorch
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
# Fallback to CPU for unsupported MPS ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def load_documents(file_path: str, ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Load and preprocess documents from file."""
    logger.info("Loading documents from %s", file_path)
    file_path_str = str(file_path)
    docs = load_file(file_path_str)
    documents = []

    if ids is not None and len(ids) != len(docs):
        logger.error("Provided IDs length (%d) does not match documents length (%d)", len(
            ids), len(docs))
        raise ValueError("IDs length must match documents length")

    for idx, doc in enumerate(tqdm(docs, desc="Processing documents")):
        doc_id = ids[idx] if ids else str(uuid.uuid4())
        if isinstance(doc, dict) and "metadata" in doc and "header_level" in doc["metadata"]:
            if doc["metadata"].get("header_level") != 1:
                text = "\n".join([
                    doc["metadata"].get("parent_header", ""),
                    doc["metadata"].get("header", ""),
                    doc["metadata"].get("content", "")
                ]).strip()
                documents.append({"text": text, "id": doc_id, "index": idx})
        else:
            text = doc.get("text", "") if isinstance(doc, dict) else ""
            logger.warning(
                "Document %s lacks metadata, using text: %s", doc_id, text[:50])
            documents.append({"text": text, "id": doc_id, "index": idx})
    return documents


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


def get_original_document(doc_id: str, documents: List[Dict[str, Any]]) -> str:
    """Retrieve original document text by ID."""
    logger.info("Retrieving original document for ID %s", doc_id)
    for doc in documents:
        if doc["id"] == doc_id:
            return doc["text"]
    return None


def search_docs(
    file_path: str,
    query: str,
    ids: Optional[List[str]] = None,
    embedder_model: str = "static-retrieval-mrl-en-v1",
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    chunk_size: int = 800,
    overlap: int = 200,
    top_k: int = 20,
    rerank_top_k: int = 5,
    batch_size: int = 8,
    bm25_weight: float = 0.5
) -> List[Dict[str, Any]]:
    """Search documents using hybrid retrieval with BM25 and embeddings."""
    total_start_time = time.time()
    documents = load_documents(file_path, ids)

    # Splitting documents
    start_time = time.time()
    chunks = []
    for doc in tqdm(documents, desc="Splitting documents"):
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
        embedder_model, device="cpu", backend="onnx")  # Use ONNX on CPU
    cross_encoder = CrossEncoder(cross_encoder_model)
    chunk_embeddings = embed_chunks_parallel(chunk_texts, embedder)

    # FAISS search
    start_time = time.time()
    dim = chunk_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Use inner product for cosine similarity
    faiss.normalize_L2(chunk_embeddings)  # Normalize for cosine similarity
    index.add(chunk_embeddings)
    top_k = min(top_k, len(chunk_texts))
    logger.info("Performing FAISS search with top-k=%d", top_k)
    query_embedding = embedder.encode([query], convert_to_numpy=True)
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
    pairs = [[query, doc["text"]] for doc, _, _ in initial_docs]
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
    results = []
    seen_doc_ids = set()
    for i, (chunk, combined_score, rerank_score, embedding_score) in enumerate(reranked_docs):
        doc_id = chunk["doc_id"]
        if doc_id not in seen_doc_ids:
            original_doc = get_original_document(doc_id, documents)
            result = {
                "id": doc_id,
                "doc_index": chunk["doc_index"],
                "rank": i + 1,
                "score": rerank_score,
                "combined_score": combined_score,
                "embedding_score": embedding_score,
                "headers": chunk["headers"],
                "text": original_doc if original_doc else "Not found"
            }
            results.append(result)
            seen_doc_ids.add(doc_id)

    total_duration = time.time() - total_start_time
    logger.info("Total search completed in %.3f seconds", total_duration)
    return results
