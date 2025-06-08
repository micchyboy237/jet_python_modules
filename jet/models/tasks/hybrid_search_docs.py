import os
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import torch
from concurrent.futures import ThreadPoolExecutor
import re
from tqdm import tqdm
from jet.file.utils import load_file
from jet.logger import logger


def load_documents(file_path):
    logger.info("Loading documents from %s", file_path)
    docs = load_file(file_path)
    documents = [
        {
            "text": "\n".join([
                doc["metadata"].get("parent_header") or "",
                doc["metadata"]["header"],
                doc["metadata"]["content"]
            ]).strip(),
            "id": idx
        }
        for idx, doc in enumerate(tqdm(docs, desc="Processing documents"))
        if doc["metadata"]["header_level"] != 1
    ]
    return documents


def split_document(doc_text, doc_id, chunk_size=800, overlap=200):
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


def filter_by_headers(chunks, query):
    logger.info("Filtering chunks by headers for query: %s", query)
    query_terms = set(query.lower().split())
    filtered = []
    for chunk in tqdm(chunks, desc="Filtering chunks"):
        headers = [h.lower() for h in chunk["headers"]]
        if any(any(term in h for term in query_terms) for h in headers) or not headers:
            filtered.append(chunk)
    return filtered if filtered else chunks


def embed_chunk(chunk, embedder):
    return embedder.encode(chunk, convert_to_numpy=True)


def embed_chunks_parallel(chunk_texts, embedder):
    logger.info("Embedding %d chunks in parallel", len(chunk_texts))
    with ThreadPoolExecutor() as executor:
        embeddings = list(tqdm(
            executor.map(lambda x: embed_chunk(x, embedder), chunk_texts),
            total=len(chunk_texts),
            desc="Embedding chunks"
        ))
    return np.vstack(embeddings)


def get_original_document(doc_id, documents):
    logger.info("Retrieving original document for ID %d", doc_id)
    for doc in documents:
        if doc["id"] == doc_id:
            return doc["text"]
    return None


def search_docs(
    file_path,
    query,
    embedder_model="all-MiniLM-L12-v2",
    cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    chunk_size=800,
    overlap=200,
    top_k=20,
    rerank_top_k=5,
    batch_size=8
):
    # Load documents
    documents = load_documents(file_path)

    # Split documents into chunks
    logger.info("Splitting %d documents into chunks", len(documents))
    chunks = []
    for doc in tqdm(documents, desc="Splitting documents"):
        chunks.extend(split_document(
            doc["text"], doc["id"], chunk_size, overlap))

    # Filter chunks
    filtered_chunks = filter_by_headers(chunks, query)
    chunk_texts = [chunk["text"] for chunk in filtered_chunks]
    logger.info("Filtered to %d chunks", len(chunk_texts))

    # Initialize models
    logger.info("Initializing SentenceTransformer and CrossEncoder models")
    embedder = SentenceTransformer(embedder_model)
    cross_encoder = CrossEncoder(cross_encoder_model)

    # Embed chunks
    chunk_embeddings = embed_chunks_parallel(chunk_texts, embedder)

    # FAISS index
    logger.info("Building FAISS index")
    dim = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(chunk_embeddings)

    # Initial retrieval
    top_k = top_k if len(chunk_texts) < 1000 else 50
    logger.info("Performing FAISS search with top-k=%d", top_k)
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    embed_scores = [1 / (1 + d) for d in distances[0]]
    initial_docs = [
        (filtered_chunks[i], embed_scores[j])
        for j, i in enumerate(indices[0])
    ]
    logger.debug("FAISS search results: indices=%s, embed_scores=%s",
                 indices[0][:5], embed_scores[:5])

    # Cross-encoder reranking
    logger.info("Reranking %d documents with cross-encoder", len(initial_docs))
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
    logger.debug("All rerank scores: %s", scores[:5])
    reranked_indices = np.argsort(scores)[::-1][:rerank_top_k]
    reranked_docs = [
        (initial_docs[i][0], initial_docs[i][1], scores[i])
        for i in reranked_indices
    ]
    logger.debug("Top %d reranked docs: %s", rerank_top_k, [
                 (doc[0]["doc_id"], doc[1], doc[2]) for doc in reranked_docs])

    # Collect results
    results = []
    seen_doc_ids = set()
    for i, (chunk, embed_score, rerank_score) in enumerate(reranked_docs):
        doc_id = chunk["doc_id"]
        if doc_id not in seen_doc_ids:
            original_doc = get_original_document(doc_id, documents)
            result = {
                "rank": i + 1,
                "doc_id": doc_id,
                "embedding_score": embed_score,
                "rerank_score": rerank_score,
                "text": chunk["text"],
                "headers": chunk["headers"],
                "original_document": original_doc if original_doc else "Not found"
            }
            results.append(result)
            seen_doc_ids.add(doc_id)

    return results


if __name__ == "__main__":
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"
    query = "List all ongoing and upcoming isekai anime 2025."
    results = search_docs(docs_file, query)
    for result in results:
        print(f"Rank {result['rank']} (Document ID {result['doc_id']}):")
        print(f"Embedding Score: {result['embedding_score']:.4f}")
        print(f"Rerank Score: {result['rerank_score']:.4f}")
        print(f"Headers: {result['headers']}")
        print(f"Original Document:\n{result['original_document']}\n")
        print(f"Text: {result['text']}")
