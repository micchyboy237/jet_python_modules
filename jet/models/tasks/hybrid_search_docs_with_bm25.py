import string
from typing import List, Dict, Any, Optional, Tuple, TypedDict, Union
import os
import time
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from jet.file.utils import load_file
from jet.logger import logger
import re
from tqdm import tqdm
import uuid
from jet.vectors.document_types import HeaderDocument, HeaderDocumentWithScore
from jet.wordnet.spellcheck import correct_typos
from jet.wordnet.stopwords import StopWords
from jet.wordnet.words import get_words
from llama_index.core.schema import MetadataMode
from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry

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
    seen_texts = set()
    seen_ids = set()
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
                    f"{metadata["parent_header"] or ""}{" -> " if metadata["parent_header"] else ""}{metadata["header"]}",
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
        if re.match(r'^#+\s', line):
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
    query_terms = set(get_words(query.lower()))
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


def embed_chunk(chunk: str, registry: SentenceTransformerRegistry, model_id: str) -> np.ndarray:
    """Embed a single chunk using SentenceTransformerRegistry."""
    embedding = registry.generate_embeddings(
        chunk, model_id=model_id, return_format="numpy")
    return np.ascontiguousarray(embedding.astype(np.float32))


def embed_chunks_parallel(chunk_texts: List[str], registry: SentenceTransformerRegistry, model_id: str) -> np.ndarray:
    """Embed chunks in batches using SentenceTransformerRegistry."""
    start_time = time.time()
    logger.info("Embedding %d chunks in batches", len(chunk_texts))
    if not chunk_texts:
        logger.info("No chunks to embed, returning empty array")
        return np.zeros((0, 0), dtype=np.float32)
    batch_size = 32
    embeddings = registry.generate_embeddings(
        chunk_texts,
        model_id=model_id,
        batch_size=batch_size,
        show_progress=True,
        return_format="numpy"
    )
    duration = time.time() - start_time
    logger.info("Embedding completed in %.3f seconds", duration)
    logger.debug("Final embeddings shape: %s, dtype: %s",
                 embeddings.shape, embeddings.dtype)
    return embeddings


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
                             doc.id, doc.metadata.get("original_index"))
                return doc
    logger.warning("No HeaderDocument found for ID %s, index %d",
                   doc_id, doc_index)
    return None


def compute_header_scores(
    chunks: List[Dict[str, Any]],
    query: str,
    registry: SentenceTransformerRegistry,
    model_id: str
) -> List[float]:
    """Compute cosine similarities between query and concatenated headers for each chunk."""
    logger.info("Computing header similarities for %d chunks", len(chunks))
    header_texts = []
    for chunk in chunks:
        headers = chunk.get("headers", [])
        original_doc = chunk.get("original_doc")
        if original_doc and isinstance(original_doc.metadata, dict):
            parent_header = original_doc.metadata.get("parent_header", "")
            header = original_doc.metadata.get("header", "")
            header_text = f"{parent_header} -> {header}" if parent_header and header else header or parent_header
        else:
            header_text = " -> ".join(h for h in headers if h) if headers else ""
        header_texts.append(header_text or "")
    if not any(header_texts):
        logger.info("No valid headers found, returning zero similarities")
        return [0.0] * len(chunks)

    header_embeddings = embed_chunks_parallel(header_texts, registry, model_id)
    query_embedding = embed_chunk(query, registry, model_id)
    query_embedding = query_embedding.reshape(1, -1)

    faiss.normalize_L2(header_embeddings)
    faiss.normalize_L2(query_embedding)
    similarities = np.dot(header_embeddings, query_embedding.T).flatten()
    similarities = [(score + 1) / 2 for score in similarities]
    logger.debug("Header similarities: %s", similarities)
    return similarities


def compute_combined_scores(
    embed_scores: List[float],
    bm25_scores: List[float],
    header_scores: List[float],
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
        content_header_score = (embed_score + header_score) / 2
        weight = bm25_weight if with_bm25 else 0.0
        combined = weight * bm25_score + (1 - weight) * content_header_score
        combined = max(0.0, min(1.0, combined))
        logger.debug(
            "Chunk %d (FAISS index %d): BM25=%.4f, Embed=%.4f, Header=%.4f, Combined=%.4f",
            i, j, bm25_score, embed_score, header_score, combined
        )
    return combined_scores


def highlight_text(text: str, query: str) -> Tuple[str, List[Match]]:
    """Highlight non-stopword query terms in text and collect match details."""
    logger.debug("Highlighting text for query: %s, text: %s",
                 query, text[:100])
    if not query.strip():
        return "", []
    query_terms = query.lower().split()
    unique_terms = {
        term.strip(string.punctuation)
        for term in query_terms
        if term not in StopWords.english_stop_words
    }
    if not unique_terms:
        logger.debug(
            "All query terms are stop words or empty after filtering.")
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
    logger.debug("Final highlighted text: %s, matches: %s",
                 highlighted[:100], matches)
    return highlighted, matches


def search_docs(
    query: str,
    documents: Union[List[HeaderDocument], List[Dict[str, Any]], List[str]],
    instruction: Optional[str] = None,
    ids: Optional[List[str]] = None,
    model_id: str = "static-retrieval-mrl-en-v1",
    chunk_size: int = 800,
    overlap: int = 200,
    top_k: Optional[int] = 20,
    threshold: float = 0.0,
    filter_by_headers_enabled: bool = True,
    bm25_weight: float = 0.3,
    with_bm25: bool = False
) -> List[HeaderDocumentWithScore]:
    total_start_time = time.time()
    logger.debug("Input document IDs: %s", [
        doc.id if isinstance(doc, HeaderDocument) else doc.get(
            "id") if isinstance(doc, dict) else str(uuid.uuid4())
        for doc in documents
        if doc
    ])

    if instruction and isinstance(instruction, str) and instruction.strip():
        logger.info("Using instruction: %s", instruction)
        query_with_instruction = f"{instruction} {query}".strip()
    else:
        logger.warning("Invalid or empty instruction, using query alone")
        query_with_instruction = query

    docs = process_documents(documents, ids)

    start_time = time.time()
    chunks = []
    for doc in tqdm(docs, desc="Splitting documents"):
        chunks.extend(split_document(
            doc["text"], doc["id"], doc["index"], chunk_size, overlap))
    logger.info("Document splitting completed in %.3f seconds, created %d chunks",
                time.time() - start_time, len(chunks))

    filtered_chunks = filter_by_headers(
        chunks, query) if filter_by_headers_enabled else chunks
    chunk_texts = [chunk["text"] for chunk in filtered_chunks]

    logger.debug("Filtered chunks: %s", [
        {"index": i, "doc_id": chunk["doc_id"],
         "text": chunk["text"][:50] + "..."}
        for i, chunk in enumerate(filtered_chunks)
    ])

    logger.info("Initializing SentenceTransformerRegistry")
    registry = SentenceTransformerRegistry()

    chunk_embeddings = embed_chunks_parallel(chunk_texts, registry, model_id)
    header_scores = compute_header_scores(
        filtered_chunks, query_with_instruction, registry, model_id)

    bm25_scores = []
    if with_bm25:
        tokenized_chunks = [chunk["text"].lower().split()
                            for chunk in filtered_chunks]
        bm25 = BM25Okapi(tokenized_chunks)
        tokenized_query = query_with_instruction.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query).tolist()
        max_bm25 = max(bm25_scores) if bm25_scores else 1.0
        bm25_scores = [score / max_bm25 if max_bm25 >
                       0 else 0.0 for score in bm25_scores]

    top_k = min(top_k or len(chunk_texts), len(chunk_texts))
    logger.info("Performing FAISS search with top-k=%d", top_k)

    dim = chunk_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(chunk_embeddings)
    index.add(chunk_embeddings)

    query_embedding = embed_chunk(query_with_instruction, registry, model_id)
    query_embedding = query_embedding.reshape(1, -1)
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, top_k)
    embed_scores = [(score + 1) / 2 for score in distances[0].tolist()]

    logger.debug("Normalized embed scores: %s", embed_scores)

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
            logger.debug(
                "Skipping doc_id %s at rank %d: combined score %.4f below threshold %.4f",
                chunk["doc_id"], rank, combined_score, threshold)
            continue
        doc_id = chunk["doc_id"]
        doc_index = chunk["doc_index"]
        if doc_id in seen_doc_ids:
            logger.debug(
                "Skipping duplicate doc_id %s at rank %d", doc_id, rank)
            continue
        original_doc = get_original_document(doc_id, doc_index, documents, ids)
        if original_doc is None:
            logger.warning(
                "Skipping doc_id %s: original document not found", doc_id)
            continue
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
        logger.debug("Added result for doc_id %s: combined=%.4f, embed=%.4f",
                     doc_id, combined_score, embedding_score)

    logger.info("Total unique results: %d", len(results))
    logger.info("Search completed in %.3f seconds",
                time.time() - total_start_time)

    return results
