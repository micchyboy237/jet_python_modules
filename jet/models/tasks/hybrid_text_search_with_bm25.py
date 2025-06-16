from typing import List, Dict, Any, Optional, TypedDict, Union, Set
import os
import time
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from rank_bm25 import BM25Okapi
from jet.logger import logger
import re
from tqdm import tqdm
import uuid
from jet.vectors.document_types import HeaderDocument
from jet.wordnet.spellcheck import correct_typos

import html
from cachetools import LRUCache

from jet.wordnet.words import get_words
import string
from nltk.corpus import stopwords
from nltk import download as nltk_download

# Ensure stopwords are available
try:
    STOP_WORDS: Set[str] = set(stopwords.words('english'))
except LookupError:
    nltk_download('stopwords')
    STOP_WORDS: Set[str] = set(stopwords.words('english'))

# Set environment variables for Mac M1 compatibility
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

MODEL_CACHE = LRUCache(maxsize=2)


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
    highlighted_text: str
    metadata: Dict[str, Any]


def process_documents(
    documents: Union[List[HeaderDocument], List[Dict[str, Any]], List[str]],
    ids: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Process and preprocess HeaderDocument objects, dictionaries, or strings."""
    logger.info("Processing %d input objects",
                len(documents) if documents else 0)
    result = []
    processed_docs: List[HeaderDocument] = []
    seen_texts = set()
    seen_ids = set()

    if documents is None:
        logger.error("Received None as documents input")
        raise ValueError("Documents input cannot be None")

    if not isinstance(documents, list):
        logger.error("Input must be a list, got %s", type(documents))
        raise ValueError("Input must be a list")

    if documents and isinstance(documents[0], str):
        for idx, text in enumerate(tqdm(documents, desc="Converting strings")):
            if text in seen_texts:
                logger.debug(
                    "Skipping duplicate text at index %d: %s", idx, text[:50])
                continue
            doc_id = ids[idx] if ids and idx < len(
                ids) else f"doc_{uuid.uuid4()}"
            if doc_id in seen_ids:
                doc_id = f"doc_{uuid.uuid4()}"
            header_doc = HeaderDocument(
                id=doc_id,
                text=text,
                metadata={"original_index": idx, "category": "default"}
            )
            processed_docs.append(header_doc)
            seen_texts.add(text)
            seen_ids.add(doc_id)
    elif documents and isinstance(documents[0], dict):
        for idx, doc_dict in enumerate(tqdm(documents, desc="Converting dictionaries")):
            header_doc = HeaderDocument(**doc_dict)
            if header_doc.text in seen_texts:
                continue
            doc_id = header_doc.id
            if doc_id in seen_ids:
                doc_id = f"doc_{uuid.uuid4()}"
                header_doc.id = doc_id
            header_doc.metadata["original_index"] = idx
            processed_docs.append(header_doc)
            seen_texts.add(header_doc.text)
            seen_ids.add(doc_id)
    else:
        for idx, doc in enumerate(tqdm(documents, desc="Processing HeaderDocuments")):
            if doc.text in seen_texts:
                continue
            doc_id = doc.id
            if doc_id in seen_ids:
                doc_id = f"doc_{uuid.uuid4()}"
                doc.id = doc_id
            doc.metadata["original_index"] = idx
            processed_docs.append(doc)
            seen_texts.add(doc.text)
            seen_ids.add(doc_id)

    for idx, doc in enumerate(tqdm(processed_docs, desc="Finalizing documents")):
        doc_id = doc.id
        text = doc.text or ""
        metadata = doc.metadata or {}
        if "header_level" in metadata and metadata.get("header_level") != 1:
            text = "\n".join([
                metadata.get("parent_header") or "",
                metadata.get("header", ""),
                metadata.get("content", "")
            ]).strip() or text
        result.append({"text": text, "id": doc_id,
                      "index": idx, "metadata": metadata})
    logger.info("Processed %d unique documents", len(result))
    return result


def highlight_text(text: str, query: str) -> str:
    """Highlight non-stopword query terms in text, removing highlights at edges if they are stop words."""
    logger.debug("Highlighting text for query: %s, text: %s",
                 query, text[:100])
    if not query.strip():
        return ""

    query_terms = query.lower().split()
    unique_terms = {term.strip(string.punctuation)
                    for term in query_terms if term not in STOP_WORDS}

    if not unique_terms:
        logger.debug(
            "All query terms are stop words or empty after filtering.")
        return ""

    def highlight_match(match: re.Match) -> str:
        word = match.group(0)
        lower_word = word.lower().strip(string.punctuation)
        if lower_word in unique_terms:
            return f"<mark>{word}</mark>"
        return word

    pattern = re.compile(
        r'\b(' + '|'.join(map(re.escape, unique_terms)) + r')\b', re.IGNORECASE)
    highlighted = pattern.sub(highlight_match, text)

    logger.debug("Final highlighted text: %s", highlighted[:100])
    return highlighted


def split_document(doc_text: str, doc_id: str, doc_index: int, metadata: Dict[str, Any], chunk_size: int = 800, overlap: int = 200) -> List[Dict[str, Any]]:
    """Split document into semantic chunks."""
    logger.info("Splitting document ID %s (index %d)", doc_id, doc_index)
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
            if current_chunk or current_content:
                chunk_text = "\n".join(current_chunk + current_content).strip()
                chunks.append({
                    "text": chunk_text,
                    "headers": current_headers.copy(),
                    "doc_id": doc_id,
                    "doc_index": doc_index,
                    "metadata": metadata
                })
                current_content = current_content[-int(
                    overlap / 2):] if overlap else []
                current_chunk = current_chunk if overlap else []
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
                if (current_chunk or current_content) and current_len + sentence_len > chunk_size:
                    chunk_text = "\n".join(
                        current_chunk + current_content).strip()
                    chunks.append({
                        "text": chunk_text,
                        "headers": current_headers.copy(),
                        "doc_id": doc_id,
                        "doc_index": doc_index,
                        "metadata": metadata
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
            chunks.append({
                "text": chunk_text,
                "headers": current_headers.copy(),
                "doc_id": doc_id,
                "doc_index": doc_index,
                "metadata": metadata
            })
    logger.info("Created %d chunks for document ID %s", len(chunks), doc_id)
    return chunks


def filter_by_headers_and_facets(chunks: List[Dict[str, Any]], query: str, facets: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Filter chunks by headers, text content, and metadata facets."""
    start_time = time.time()
    logger.info("Filtering %d chunks for query: %s, facets: %s",
                len(chunks), query, facets)
    query_terms = set(get_words(query.lower()))
    filtered = []

    for chunk in tqdm(chunks, desc="Filtering chunks"):
        headers = [h.lower() for h in chunk["headers"]]
        text = chunk["text"].lower()
        metadata = chunk.get("metadata", {})
        include = any(any(term in h for term in query_terms)
                      for h in headers) or any(term in text for term in query_terms)
        logger.debug("Chunk doc_id %s, headers: %s, text: %s, include: %s",
                     chunk["doc_id"], headers, text[:50], include)

        if facets:
            for key, value in facets.items():
                if metadata.get(key) != value:
                    include = False
                    logger.debug(
                        "Excluding chunk doc_id %s due to facet mismatch: %s=%s", chunk["doc_id"], key, value)
                    break
        if include:
            filtered.append(chunk)

    duration = time.time() - start_time
    logger.info("Filtering completed in %.3f seconds, reduced %d to %d chunks",
                duration, len(chunks), len(filtered))
    return filtered


def embed_chunks_parallel(chunk_texts: List[str], embedder: SentenceTransformer) -> np.ndarray:
    """Embed chunks in batches."""
    start_time = time.time()
    logger.info("Embedding %d chunks in batches", len(chunk_texts))
    if not chunk_texts:
        return np.zeros((0, embedder.get_sentence_embedding_dimension()), dtype=np.float32)
    batch_size = 32
    embeddings = []
    for i in tqdm(range(0, len(chunk_texts), batch_size), desc="Embedding chunks"):
        batch = chunk_texts[i:i + batch_size]
        batch_embeddings = embedder.encode(
            batch, convert_to_numpy=True, batch_size=batch_size)
        batch_embeddings = np.ascontiguousarray(
            batch_embeddings.astype(np.float32))
        embeddings.extend(batch_embeddings)
    duration = time.time() - start_time
    logger.info("Embedding completed in %.3f seconds", duration)
    return np.vstack(embeddings)


def get_bm25_scores(chunk_texts: List[str], query: str, typo_tolerance: bool = True) -> List[float]:
    """Calculate BM25 scores with typo tolerance, handling small corpora."""
    logger.info("Calculating BM25 scores for %d chunks, query: %s, typo_tolerance: %s",
                len(chunk_texts), query, typo_tolerance)
    if not chunk_texts or not query.strip():
        logger.warning("Empty chunk_texts or query, returning zero scores")
        return [0.0] * len(chunk_texts)

    tokenized_chunks = []
    for i, text in enumerate(chunk_texts):
        tokens = get_words(text.lower())
        if not tokens:
            logger.warning("Empty tokens for chunk %d, using ['empty']", i)
            tokens = ['empty']
        tokenized_chunks.append(tokens)
        logger.debug("Tokenized chunk %d: %s", i, tokens[:10])

    tokenized_query = get_words(query.lower())
    logger.debug("Original query tokens: %s", tokenized_query)

    if typo_tolerance:
        all_tokens = {t for chunk in tokenized_chunks for t in chunk}
        logger.debug("All unique tokens for typo correction: %s",
                     list(all_tokens)[:10])
        tokenized_query = correct_typos(
            query_tokens=tokenized_query,
            all_tokens=all_tokens,
        )
        logger.debug("Corrected query tokens: %s", tokenized_query)

    try:
        # Use lower k1 for small corpora, epsilon for stability
        bm25 = BM25Okapi(tokenized_chunks, k1=1.0, b=0.5, epsilon=1.0)
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
    logger.info("Retrieving document for ID %s, index %d", doc_id, doc_index)
    logger.debug("Input documents type: %s, count: %d, ids: %s",
                 type(documents), len(documents) if documents else 0, ids)

    if documents and isinstance(documents[0], str):
        for idx, text in enumerate(documents):
            expected_id = ids[idx] if ids and idx < len(ids) else f"doc_{idx}"
            logger.debug(
                "Checking string document at index %d, expected_id: %s", idx, expected_id)
            if doc_id == expected_id or (ids is None and doc_index == idx):
                header_doc = HeaderDocument(
                    id=expected_id,
                    text=text,
                    metadata={"original_index": idx}
                )
                logger.debug("Found document: ID %s, index %d",
                             expected_id, idx)
                return header_doc
    elif documents and isinstance(documents[0], dict):
        for idx, doc_dict in enumerate(documents):
            try:
                header_doc = HeaderDocument(**doc_dict)
                logger.debug(
                    "Checking dict document ID %s, index %d", header_doc.id, idx)
                if header_doc.id == doc_id or (ids is None and header_doc.metadata.get("original_index") == doc_index):
                    logger.debug("Found document: ID %s, index %d",
                                 header_doc.id, idx)
                    return header_doc
            except Exception as e:
                logger.error(
                    "Failed to convert dict at index %d: %s", idx, str(e))
                continue
    else:
        for doc in documents:
            logger.debug("Checking HeaderDocument ID %s, index %d",
                         doc.id, doc.metadata.get("original_index"))
            if doc.id == doc_id or (ids is None and doc.metadata.get("original_index") == doc_index):
                logger.debug("Found document: ID %s, index %d",
                             doc.id, doc.metadata.get("original_index"))
                return doc
    logger.warning("No document found for ID %s, index %d", doc_id, doc_index)
    return None


def search_texts(
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
    bm25_weight: float = 0.5,
    facets: Optional[Dict[str, Any]] = None,
    typo_tolerance: bool = True
) -> List[SearchResult]:
    """Search documents with Algolia-like features."""
    total_start_time = time.time()
    logger.info("Starting search with query: %s, facets: %s, typo_tolerance: %s",
                query, facets, typo_tolerance)

    docs = process_documents(documents, ids)
    chunks = []
    for doc in tqdm(docs, desc="Splitting documents"):
        chunks.extend(split_document(
            doc["text"], doc["id"], doc["index"], doc["metadata"], chunk_size, overlap))

    filtered_chunks = filter_by_headers_and_facets(chunks, query, facets)
    if not filtered_chunks:
        logger.warning("No chunks after filtering, returning empty results")
        return []

    chunk_texts = [chunk["text"] for chunk in filtered_chunks]
    logger.debug("Filtered chunks: %s", [
                 {"doc_id": c["doc_id"], "text": c["text"][:50]} for c in filtered_chunks])

    embedder_key = f"embedder_{model}"
    cross_encoder_key = f"cross_encoder_{rerank_model}"
    if embedder_key not in MODEL_CACHE:
        MODEL_CACHE[embedder_key] = SentenceTransformer(
            model, device="cpu", backend="onnx")
    if cross_encoder_key not in MODEL_CACHE:
        MODEL_CACHE[cross_encoder_key] = CrossEncoder(
            rerank_model, device="cpu", backend="onnx")
    embedder = MODEL_CACHE[embedder_key]
    cross_encoder = MODEL_CACHE[cross_encoder_key]

    chunk_embeddings = embed_chunks_parallel(chunk_texts, embedder)
    dim = chunk_embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = 40
    index.hnsw.efSearch = 16
    faiss.normalize_L2(chunk_embeddings)
    index.add(chunk_embeddings)

    top_k = min(top_k or len(chunk_texts), len(chunk_texts))
    logger.debug("Adjusted top_k: %d", top_k)
    query_with_instruction = f"{instruction} {query}".strip(
    ) if instruction else query
    query_embedding = embedder.encode(
        [query_with_instruction], convert_to_numpy=True)
    query_embedding = np.ascontiguousarray(query_embedding.astype(np.float32))
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, top_k)
    embed_scores = [(score + 1) / 2 for score in distances[0].tolist()]
    logger.debug("FAISS distances: %s, indices: %s",
                 distances[0].tolist(), indices[0].tolist())

    bm25_scores = get_bm25_scores(chunk_texts, query, typo_tolerance)
    max_bm25 = max(bm25_scores) if bm25_scores and max(
        bm25_scores) > 0 else 1.0
    bm25_scores = [score / max_bm25 for score in bm25_scores]
    logger.debug("Normalized BM25 scores: %s", bm25_scores)

    combined_scores = [
        bm25_weight * bm25_scores[i] + (1 - bm25_weight) * embed_scores[j]
        for j, i in enumerate(indices[0])
    ]

    initial_docs = [
        (filtered_chunks[i], combined_scores[j], embed_scores[j])
        for j, i in enumerate(indices[0])
    ]

    pairs = [[query_with_instruction, doc["text"]]
             for doc, _, _ in initial_docs]
    scores = []
    for i in tqdm(range(0, len(pairs), batch_size), desc="Reranking"):
        batch = pairs[i:i + batch_size]
        batch_scores = cross_encoder.predict(batch)
        scores.extend(batch_scores)

    reranked_indices = np.argsort(
        scores)[::-1][:rerank_top_k or len(initial_docs)]
    reranked_docs = [
        (initial_docs[i][0], initial_docs[i][1], scores[i], initial_docs[i][2])
        for i in reranked_indices
    ]

    results: List[SearchResult] = []
    seen_doc_ids = set()
    for i, (chunk, combined_score, rerank_score, embedding_score) in enumerate(reranked_docs):
        doc_id = chunk["doc_id"]
        doc_index = chunk["doc_index"]
        if doc_id in seen_doc_ids:
            continue
        original_doc = get_original_document(doc_id, doc_index, documents, ids)
        if original_doc is None:
            continue
        results.append({
            "id": doc_id,
            "doc_index": doc_index,
            "rank": len(results) + 1,
            "score": rerank_score,
            "combined_score": combined_score,
            "embedding_score": embedding_score,
            "headers": chunk["headers"],
            "text": original_doc.text,
            "highlighted_text": highlight_text(original_doc.text, query),
            "document": original_doc,
            "metadata": chunk.get("metadata", {})
        })
        seen_doc_ids.add(doc_id)
        if len(results) >= rerank_top_k:
            break

    total_duration = time.time() - total_start_time
    logger.info("Search completed in %.3f seconds, returned %d results",
                total_duration, len(results))
    return results
