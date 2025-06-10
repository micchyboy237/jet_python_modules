import os
import uuid
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from jet.file.utils import load_file
from jet.logger import logger
from jet.vectors.document_types import HeaderDocument
from jet.models.tasks.task_types import SimilarityResult

# Set environment variables before importing numpy/pytorch
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
# Fallback to CPU for unsupported MPS ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Initialize global models
embedder = None
cross_encoder = None


def initialize_models(model: str, rerank_model: str) -> None:
    """Initialize SentenceTransformer and CrossEncoder models."""
    global embedder, cross_encoder
    if embedder is None or cross_encoder is None:
        logger.info(
            "Initializing SentenceTransformer (%s) and CrossEncoder (%s)", model, rerank_model)
        embedder = SentenceTransformer(model, device="cpu", backend="onnx")
        cross_encoder = CrossEncoder(rerank_model)


def load_documents(file_path: str) -> List[dict]:
    """Load and process documents from a file."""
    logger.info("Loading documents from %s", file_path)
    docs = load_file(file_path)
    documents = [
        {
            "text": "\n".join([
                doc["metadata"].get("parent_header", ""),
                doc["metadata"]["header"],
                doc["metadata"]["content"]
            ]).strip(),
            "id": idx,
            "metadata": doc["metadata"]
        }
        for idx, doc in enumerate(tqdm(docs, desc="Processing documents"))
        if doc["metadata"]["header_level"] != 1
    ]
    return documents


def split_document(doc_text: str, doc_id: int, chunk_size: int = 800, overlap: int = 200) -> List[dict]:
    """Split a document into chunks with headers and metadata."""
    logger.info("Splitting document ID %d into chunks", doc_id)
    chunks = []
    headers = []
    lines = doc_text.split("\n")
    current_chunk = ""
    current_len = 0
    for line in lines:
        if line.startswith(("# ", "## ")):
            headers.append(line)
        line_len = len(line.split())
        if line.startswith(("# ", "## ")) or current_len + line_len > chunk_size:
            if current_chunk:
                chunks.append({
                    "text": current_chunk.strip(),
                    "headers": headers.copy(),
                    "doc_id": doc_id
                })
            if line.startswith(("# ", "## ")):
                current_chunk = line
                current_len = line_len
            else:
                current_chunk = line
                current_len = line_len
        else:
            current_chunk += "\n" + line
            current_len += line_len
    if current_chunk:
        chunks.append({
            "text": current_chunk.strip(),
            "headers": headers.copy(),
            "doc_id": doc_id
        })
    return chunks


def filter_by_headers(chunks: List[dict], query: str) -> List[dict]:
    """Filter chunks based on query relevance to headers."""
    logger.info("Filtering chunks by headers for query: %s", query)
    query_terms = set(query.lower().split())
    filtered = []
    for chunk in tqdm(chunks, desc="Filtering chunks"):
        headers = [h.lower() for h in chunk["headers"]]
        if any(any(term in h for term in query_terms) for h in headers) or not headers:
            filtered.append(chunk)
    return filtered if filtered else chunks


def embed_chunk(chunk: str) -> np.ndarray:
    """Embed a single chunk using the SentenceTransformer model."""
    return embedder.encode(chunk, convert_to_numpy=True)


def embed_chunks_parallel(chunk_texts: List[str]) -> np.ndarray:
    """Embed chunks in parallel using ThreadPoolExecutor."""
    logger.info("Embedding %d chunks in parallel", len(chunk_texts))
    with ThreadPoolExecutor() as executor:
        embeddings = list(tqdm(
            executor.map(embed_chunk, chunk_texts),
            total=len(chunk_texts),
            desc="Embedding chunks"
        ))
    return np.vstack(embeddings)


def get_original_document(doc_id: int, documents: List[dict]) -> Optional[dict]:
    """Retrieve the original document by ID."""
    logger.info("Retrieving original document for ID %s", doc_id)
    for doc in documents:
        if doc["id"] == doc_id:
            return doc
    logger.warning("Original document not found for ID %s", doc_id)
    return None


def search_docs(
    query: str,
    documents: List[HeaderDocument],
    task_description: str,
    model: str = "static-retrieval-mrl-en-v1",
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    max_length: int = 512,
    ids: List[str] = [],
    threshold: float = 0.0
) -> List[SimilarityResult]:
    """
    Search documents using embeddings and reranking.

    Args:
        query: The search query string.
        documents: List of HeaderDocument objects to search through.
        task_description: Description of the search task for context.
        model: SentenceTransformer model name for embeddings.
        rerank_model: CrossEncoder model name for reranking.
        max_length: Maximum length for chunk texts.
        ids: Optional list of document IDs to filter results.
        threshold: Minimum similarity score threshold.

    Returns:
        List of SimilarityResult objects with ranked documents.
    """
    logger.info("Starting document search for query: %s, task: %s",
                query, task_description)

    # Initialize models
    initialize_models(model, rerank_model)

    # Convert HeaderDocument to internal format, preserving original index
    internal_docs = [
        {
            "text": doc.get_recursive_text(),
            "id": doc.id or str(uuid.uuid4()),
            "metadata": doc.metadata,
            "original_index": idx  # Store original index from input list
        }
        for idx, doc in enumerate(documents)
    ]
    logger.debug("Created %d internal documents: %s", len(internal_docs),
                 [doc["id"] for doc in internal_docs[:5]])

    # Filter documents by provided IDs if any
    if ids:
        internal_docs = [doc for doc in internal_docs if doc["id"] in ids]
        logger.info("Filtered to %d documents based on provided IDs",
                    len(internal_docs))

    # Split documents into chunks
    logger.info("Splitting %d documents into chunks", len(internal_docs))
    chunks = []
    for doc in tqdm(internal_docs, desc="Splitting documents"):
        chunks.extend(split_document(doc["text"], doc["id"]))

    # Filter chunks by headers
    filtered_chunks = filter_by_headers(chunks, query)
    chunk_texts = [chunk["text"][:max_length] for chunk in filtered_chunks]
    logger.info("Filtered to %d chunks", len(chunk_texts))

    # Embed chunks
    chunk_embeddings = embed_chunks_parallel(chunk_texts)

    # FAISS index
    logger.info("Building FAISS index")
    dim = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(chunk_embeddings)

    # Initial retrieval
    k = min(20 if len(chunk_texts) < 1000 else 50, len(chunk_texts))
    logger.info("Performing FAISS search with top-k=%d", k)
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    embed_scores = [1 / (1 + d) for d in distances[0]]
    initial_docs = [
        (filtered_chunks[i], embed_scores[j])
        for j, i in enumerate(indices[0])
    ]
    logger.debug("FAISS search results: indices=%s, embed_scores=%s",
                 indices[0][:5], embed_scores[:5])

    # Cross-encoder reranking
    logger.info("Reranking %d documents with cross-encoder", len(initial_docs))
    batch_size = 8
    pairs = [[query, doc["text"]] for doc, _ in initial_docs]
    scores = []
    try:
        for i in tqdm(range(0, len(pairs), batch_size), desc="Reranking batches"):
            batch = pairs[i:i + batch_size]
            batch_scores = cross_encoder.predict(batch)
            scores.extend(batch_scores)
            logger.debug("Batch %d rerank scores: %s",
                         i // batch_size, batch_scores)
    except Exception as e:
        logger.error("Error in reranking: %s", e)
        scores = [0] * len(pairs)  # Fallback
    logger.debug("All rerank scores: %s", scores[:10])

    # Prepare results
    reranked_indices = np.argsort(scores)[::-1][:10]
    seen_doc_ids = set()
    results: List[SimilarityResult] = []
    rank = 1
    for i in reranked_indices:
        chunk, embed_score, rerank_score = initial_docs[i][0], initial_docs[i][1], scores[i]
        doc_id = chunk["doc_id"]
        if doc_id not in seen_doc_ids and rerank_score >= threshold:
            original_doc = get_original_document(doc_id, internal_docs)
            if original_doc:
                result: SimilarityResult = {
                    "id": str(doc_id),
                    "rank": rank,
                    # Use original index
                    "doc_index": int(original_doc["original_index"]),
                    "score": float(rerank_score),
                    "text": original_doc["text"][:max_length],
                    "tokens": len(original_doc["text"].split())
                }
                results.append(result)
                seen_doc_ids.add(doc_id)
                rank += 1
            else:
                logger.warning("Original document not found for ID %s", doc_id)
    logger.debug("Final results: %s", [
        {"id": r["id"], "doc_index": r["doc_index"], "rank": r["rank"]}
        for r in results
    ])

    logger.info("Returning %d ranked results", len(results))
    return results
