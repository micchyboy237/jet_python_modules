import os
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from rank_bm25 import BM25Okapi
from concurrent.futures import ThreadPoolExecutor
from jet.file.utils import load_file
from jet.logger import logger
from typing import List, Dict, Any
import re
from tqdm import tqdm

# Set environment variables before importing numpy/pytorch
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"


def load_documents(file_path: str) -> List[Dict[str, Any]]:
    """Load and preprocess documents from file."""
    logger.info("Loading documents from %s", file_path)
    file_path_str = str(file_path)
    docs = load_file(file_path_str)
    documents = []
    for idx, doc in enumerate(tqdm(docs, desc="Processing documents")):
        if isinstance(doc, dict) and "metadata" in doc and "header_level" in doc["metadata"]:
            if doc["metadata"].get("header_level") != 1:
                text = "\n".join([
                    doc["metadata"].get("parent_header", ""),
                    doc["metadata"].get("header", ""),
                    doc["metadata"].get("content", "")
                ]).strip()
                documents.append({"text": text, "id": idx})
        else:
            text = doc.get("text", "") if isinstance(doc, dict) else ""
            logger.warning(
                "Document %d lacks metadata, using text: %s", idx, text[:50])
            documents.append({"text": text, "id": idx})
    return documents


def split_document(doc_text: str, doc_id: int, chunk_size: int = 800, overlap: int = 200) -> List[Dict[str, Any]]:
    """Split document into semantic chunks using sentence boundaries."""
    logger.info("Splitting document ID %d into chunks with chunk_size=%d, overlap=%d",
                doc_id, chunk_size, overlap)
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
                    "doc_id": doc_id
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
                        "doc_id": doc_id
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
                "doc_id": doc_id
            })

    logger.info("Created %d chunks for document ID %d", len(chunks), doc_id)
    return chunks


def filter_by_headers(chunks: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """Filter chunks based on header relevance."""
    logger.info("Filtering chunks by headers for query: %s", query)
    query_terms = set(query.lower().split())
    filtered = []
    for chunk in tqdm(chunks, desc="Filtering chunks"):
        headers = [h.lower() for h in chunk["headers"]]
        if any(any(term in h for term in query_terms) for h in headers) or not headers:
            filtered.append(chunk)
    return filtered if filtered else chunks


def embed_chunk(chunk: str, embedder: SentenceTransformer) -> np.ndarray:
    """Embed a single chunk using SentenceTransformer."""
    return embedder.encode(chunk, convert_to_numpy=True)


def embed_chunks_parallel(chunk_texts: List[str], embedder: SentenceTransformer) -> np.ndarray:
    """Embed chunks in batches to optimize performance."""
    logger.info("Embedding %d chunks in batches", len(chunk_texts))
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
    return np.vstack(embeddings)


def get_bm25_scores(chunk_texts: List[str], query: str) -> List[float]:
    """Calculate BM25 scores for chunks."""
    tokenized_chunks = [text.lower().split() for text in chunk_texts]
    bm25 = BM25Okapi(tokenized_chunks)
    tokenized_query = query.lower().split()
    return bm25.get_scores(tokenized_query).tolist()


def get_original_document(doc_id: int, documents: List[Dict[str, Any]]) -> str:
    """Retrieve original document text by ID."""
    logger.info("Retrieving original document for ID %d", doc_id)
    for doc in documents:
        if doc["id"] == doc_id:
            return doc["text"]
    return None


def search_docs(
    file_path: str,
    query: str,
    embedder_model: str = "all-MiniLM-L12-v2",
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    chunk_size: int = 800,
    overlap: int = 200,
    top_k: int = 20,
    rerank_top_k: int = 5,
    batch_size: int = 8,
    bm25_weight: float = 0.5
) -> List[Dict[str, Any]]:
    """Search documents using hybrid retrieval with BM25 and embeddings."""
    documents = load_documents(file_path)
    chunks = []
    for doc in tqdm(documents, desc="Splitting documents"):
        chunks.extend(split_document(
            doc["text"], doc["id"], chunk_size, overlap))

    filtered_chunks = filter_by_headers(chunks, query)
    chunk_texts = [chunk["text"] for chunk in filtered_chunks]
    logger.info("Filtered to %d chunks", len(chunk_texts))

    embedder = SentenceTransformer(embedder_model)
    cross_encoder = CrossEncoder(cross_encoder_model)
    chunk_embeddings = embed_chunks_parallel(chunk_texts, embedder)

    dim = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(chunk_embeddings)

    top_k = min(top_k, len(chunk_texts))
    logger.info("Performing FAISS search with top-k=%d", top_k)
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    embed_scores = [1 / (1 + d) for d in distances[0]]

    bm25_scores = get_bm25_scores(chunk_texts, query)
    combined_scores = [
        bm25_weight * bm25_scores[i] + (1 - bm25_weight) * embed_scores[j]
        for j, i in enumerate(indices[0])
    ]
    initial_docs = [
        (filtered_chunks[i], combined_scores[j])
        for j, i in enumerate(indices[0])
    ]

    logger.info("Reranking %d documents with cross-encoder", len(initial_docs))
    pairs = [[query, doc["text"]] for doc, _ in initial_docs]
    scores = []
    try:
        for i in tqdm(range(0, len(pairs), batch_size), desc="Reranking batches"):
            batch = pairs[i:i + batch_size]
            batch_scores = cross_encoder.predict(batch)
            scores.extend(batch_scores)
    except Exception as e:
        logger.error("Error in reranking: %s", e)
        scores = [0] * len(pairs)
    reranked_indices = np.argsort(scores)[::-1][:rerank_top_k]
    reranked_docs = [
        (initial_docs[i][0], initial_docs[i][1], scores[i])
        for i in reranked_indices
    ]

    results = []
    seen_doc_ids = set()
    for i, (chunk, combined_score, rerank_score) in enumerate(reranked_docs):
        doc_id = chunk["doc_id"]
        if doc_id not in seen_doc_ids:
            original_doc = get_original_document(doc_id, documents)
            result = {
                "rank": i + 1,
                "doc_id": doc_id,
                "combined_score": combined_score,
                "rerank_score": rerank_score,
                "headers": chunk["headers"],
                "text": original_doc if original_doc else "Not found"
            }
            results.append(result)
            seen_doc_ids.add(doc_id)

    return results
