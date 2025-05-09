import json
from typing import Any, Callable, Union, List, Dict, Optional, Literal, TypedDict, DefaultDict
from jet.code.utils import ProcessedResult, process_markdown_file
from jet.file.utils import load_file, save_file
from jet.llm.mlx.base import MLX
from jet.llm.mlx.mlx_types import ModelKey, ModelType
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.utils.markdown import extract_block_content
import numpy as np
from collections import defaultdict
import faiss
from sentence_transformers import SentenceTransformer
import logging
import hashlib
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from bs4 import BeautifulSoup
import trafilatura
import re
# from fast_langdetect import detect
from rank_bm25 import BM25Okapi
from nltk.tokenize import sent_tokenize, word_tokenize

MODEL: ModelKey = "qwen3-0.6b-4bit"
mlx = MLX(MODEL)


class SimilarityResult(TypedDict):
    """
    Represents a single similarity result for a text.

    Fields:
        id: Identifier for the text.
        rank: Rank based on score (1 for highest).
        doc_index: Original index of the text in the input list.
        score: Fused similarity score.
        percent_difference: Percentage difference from the highest score.
        text: The compared text (or chunk if long).
        relevance: Optional relevance score (e.g., from user feedback).
        word_count: Number of words in the text.
    """
    id: str
    rank: Optional[int]
    doc_index: int
    score: float
    percent_difference: Optional[float]
    text: str
    relevance: Optional[float]
    word_count: Optional[int]


def generate_key(text: str, query: str = None) -> str:
    """Generate a unique key for a text-query pair."""
    combined = (text + (query or "")).encode('utf-8')
    return hashlib.md5(combined).hexdigest()


@lru_cache(maxsize=1000)
def get_embedding_function(model_name: str) -> callable:
    """Load and cache a SentenceTransformer model."""
    logger.info(f"Loading model: {model_name}")
    return SentenceTransformer(model_name).encode


def preprocess_text(
    text: str,
    domain: Optional[str] = None,
    chunk_size: int = 150,
    overlap: int = 50,
    split_fn: Callable[[str], List[str]] = sent_tokenize
) -> List[str]:
    """
    Preprocesses web-scraped text, returning chunks for long texts.

    Args:
        text: Raw web-scraped text.
        domain: Optional domain (e.g., 'news', 'ecommerce') for specialized preprocessing.
        chunk_size: Maximum token length per chunk (approximate).
        overlap: Number of tokens to overlap between chunks.
        split_fn: Function to split text into logical units (default: NLTK sentence tokenizer).

    Returns:
        List of preprocessed text chunks.
    """
    # Try extracting main content with trafilatura
    extracted = None
    try:
        extracted = trafilatura.extract(
            text,
            include_comments=False,
            include_tables=False,
            no_fallback=False  # Allow fallback to simpler parsing
        )
    except Exception as e:
        logger.warning(f"Trafilatura failed to extract content: {e}")

    # Fallback to BeautifulSoup if trafilatura fails or returns None
    if not extracted:
        try:
            soup = BeautifulSoup(text, 'html.parser')
            # Remove script and style elements
            for element in soup(['nav', 'footer', 'script', 'style']):
                element.decompose()
            extracted = soup.get_text(separator=' ', strip=True)
        except Exception as e:
            logger.warning(
                f"BeautifulSoup fallback failed: {e}, using raw text")
            extracted = text  # Use raw text as last resort

    # Remove boilerplate and normalize
    text = re.sub(r'\s+', ' ', extracted.strip())  # Collapse whitespace
    text = re.sub(r'(click here|read more|sign up|log in|subscribe now)',
                  '', text, flags=re.IGNORECASE)
    text = text.lower()

    # Language filtering (keep English for simplicity)
    # try:
    #     if detect(text) != 'en':
    #         logger.warning("Non-English text detected, returning empty list")
    #         return []
    # except Exception as e:
    #     logger.warning(f"Language detection failed: {e}, proceeding with text")

    # Domain-specific preprocessing
    if domain == "news":
        text = re.sub(
            r'published on \d{4}-\d{2}-\d{2}|by [a-zA-Z\s]+', '', text)
    elif domain == "ecommerce":
        text = re.sub(r'\$\d+\.\d{2}|\d+% off', '', text)
    elif domain == "forum":
        text = re.sub(r'posted by [a-zA-Z0-9\s]+|re: ', '', text)

    # First attempt: chunk by word_tokenize on whole text
    words = word_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        current_chunk.append(word)
        current_length += 1
        if current_length >= chunk_size:
            chunks.append(' '.join(current_chunk))
            # Add overlap for the next chunk
            overlap_words = current_chunk[-overlap:] if overlap > 0 else []
            current_chunk = overlap_words
            current_length = len(overlap_words)

    # Append the last chunk if non-empty
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    # Fallback to split_fn if no chunks or text is too short
    if not chunks or len(words) < chunk_size:
        chunks = []
        units = split_fn(text)
        current_chunk = []
        current_length = 0

        for unit in units:
            unit_words = word_tokenize(unit)
            if current_length + len(unit_words) <= chunk_size:
                current_chunk.append(unit)
                current_length += len(unit_words)
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                # Start new chunk with overlap
                overlap_words = word_tokenize(
                    ' '.join(current_chunk))[-overlap:] if overlap > 0 else []
                current_chunk = [
                    ' '.join(overlap_words)] if overlap_words else []
                current_chunk.append(unit)
                current_length = len(overlap_words) + len(unit_words)

        # Append the last chunk if non-empty
        if current_chunk:
            chunks.append(' '.join(current_chunk))

    # If no chunks (e.g., empty after filtering), return single empty chunk
    return chunks if chunks else [""]


class VectorIndex:
    """Manages a FAISS index for efficient similarity search."""

    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatIP()  # Inner product for cosine similarity
        self.texts = []
        self.ids = []

    def add(self, embeddings: np.ndarray, texts: List[str], ids: List[str]):
        """Add embeddings to the index."""
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.where(norms != 0, norms, 1)
        self.index.add(embeddings)
        self.texts.extend(texts)
        self.ids.extend(ids)

    def search(self, query_embedding: np.ndarray, k: int) -> tuple:
        """Search for top-k similar texts."""
        # Normalize query embedding
        norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
        query_embedding = query_embedding / np.where(norm != 0, norm, 1)
        distances, indices = self.index.search(query_embedding, k)
        return distances, indices


def query_similarity_scores(
    query: Union[str, List[str]],
    texts: Union[str, List[str]],
    threshold: float = 0.0,
    model: Union[str, List[str]] = "all-MiniLM-L6-v2",
    fuse_method: Literal["average", "max", "min", "weighted"] = "average",
    model_weights: Optional[List[float]] = None,
    ids: Union[List[str], None] = None,
    metrics: Literal["cosine", "dot", "euclidean"] = "cosine",
    domain: Optional[str] = None,
    use_index: bool = True,
    top_k: int = 100,
    use_bm25: bool = False,
    bm25_k: int = 1000
) -> List[SimilarityResult]:
    """
    Computes similarity scores for queries against texts with scalable vector indexing and hybrid search.

    Args:
        query: Single query or list of queries.
        texts: Single text or list of texts.
        threshold: Minimum similarity score to include.
        model: One or more embedding model names.
        fuse_method: Fusion method ('average', 'max', 'min', 'weighted').
        model_weights: Weights for weighted fusion.
        ids: Optional list of IDs for texts.
        metrics: Similarity metric ('cosine', 'euclidean', 'dot').
        domain: Optional domain for preprocessing.
        use_index: Whether to use FAISS indexing.
        top_k: Number of top results to retrieve.
        use_bm25: Whether to use BM25 for initial candidate selection.
        bm25_k: Number of BM25 candidates to retrieve.

    Returns:
        List of SimilarityResult, sorted by score with ranks and metadata.
    """
    # Input normalization
    query = [query] if isinstance(query, str) else query
    texts = [texts] if isinstance(texts, str) else texts
    model = [model] if isinstance(model, str) else model

    # Validation
    if not query or not texts:
        raise ValueError("Query and texts must be non-empty.")
    if not model:
        raise ValueError("At least one model must be provided.")
    if ids is not None and len(ids) != len(texts):
        raise ValueError(
            f"Length of ids ({len(ids)}) must match texts ({len(texts)}).")
    if fuse_method == "weighted" and (not model_weights or len(model_weights) != len(model)):
        raise ValueError(
            "Model weights must match number of models for weighted fusion.")
    if fuse_method not in {"average", "max", "min", "weighted"}:
        raise ValueError(f"Unsupported fusion method: {fuse_method}")
    if metrics not in {"cosine", "dot", "euclidean"}:
        raise ValueError(f"Unsupported metrics: {metrics}")

    text_ids = ids if ids else [generate_key(text, query[0]) for text in texts]

    # Preprocess texts and queries (returns chunks for long texts)
    preprocessed_texts = [preprocess_text(text, domain) for text in texts]
    # Take first chunk for queries
    preprocessed_queries = [preprocess_text(q, domain)[0] for q in query]

    # Flatten chunks for embedding (track original document index)
    flat_texts = []
    flat_ids = []
    doc_indices = []
    for doc_idx, chunks in enumerate(preprocessed_texts):
        for chunk in chunks:
            flat_texts.append(chunk)
            flat_ids.append(text_ids[doc_idx])
            doc_indices.append(doc_idx)

    # BM25 candidate selection (optional)
    candidate_texts = flat_texts
    candidate_ids = flat_ids
    candidate_doc_indices = doc_indices
    if use_bm25 and flat_texts:
        tokenized_texts = [text.split() for text in flat_texts]
        bm25 = BM25Okapi(tokenized_texts)
        bm25_scores = bm25.get_scores(query[0].split())
        top_indices = np.argsort(bm25_scores)[::-1][:bm25_k]
        candidate_texts = [flat_texts[i] for i in top_indices]
        candidate_ids = [flat_ids[i] for i in top_indices]
        candidate_doc_indices = [doc_indices[i] for i in top_indices]

    all_results = []

    # Initialize vector index if enabled
    vector_index = None
    if use_index and candidate_texts:
        embed_func = get_embedding_function(model[0])
        sample_embedding = embed_func([candidate_texts[0]])[0]
        vector_index = VectorIndex(dimension=len(sample_embedding))
        text_embeddings = embed_func(candidate_texts)
        vector_index.add(text_embeddings, candidate_texts, candidate_ids)

    def process_model(model_name: str):
        results = []
        embed_func = get_embedding_function(model_name)

        if use_index and vector_index:
            query_embeddings = embed_func(preprocessed_queries)
            for i, q_emb in enumerate(query_embeddings):
                distances, indices = vector_index.search(
                    q_emb[np.newaxis, :], top_k)
                scores = distances[0] if metrics == "cosine" else 1 / \
                    (1 + distances[0])
                for j, idx in enumerate(indices[0]):
                    if scores[j] >= threshold:
                        results.append({
                            "id": vector_index.ids[idx],
                            "doc_index": candidate_doc_indices[idx],
                            "query": query[i],
                            "text": vector_index.texts[idx],
                            "score": float(scores[j])
                        })
        else:
            query_embeddings = embed_func(preprocessed_queries)
            text_embeddings = embed_func(candidate_texts)

            if metrics == "cosine":
                query_norms = np.linalg.norm(
                    query_embeddings, axis=1, keepdims=True)
                text_norms = np.linalg.norm(
                    text_embeddings, axis=1, keepdims=True)
                query_embeddings /= np.where(query_norms != 0, query_norms, 1)
                text_embeddings /= np.where(text_norms != 0, text_norms, 1)
                similarity_matrix = np.dot(query_embeddings, text_embeddings.T)
            elif metrics == "dot":
                similarity_matrix = np.dot(query_embeddings, text_embeddings.T)
            elif metrics == "euclidean":
                similarity_matrix = np.zeros(
                    (len(query), len(candidate_texts)))
                for i in range(len(query)):
                    for j in range(len(candidate_texts)):
                        dist = np.linalg.norm(
                            query_embeddings[i] - text_embeddings[j])
                        similarity_matrix[i, j] = 1 / (1 + dist)

            for i, q in enumerate(query):
                scores = similarity_matrix[i]
                mask = scores >= threshold
                filtered_indices = np.arange(len(candidate_texts))[mask]
                filtered_scores = scores[mask]
                sorted_indices = np.argsort(filtered_scores)[::-1]
                for idx, j in enumerate(sorted_indices):
                    results.append({
                        "id": candidate_ids[filtered_indices[j]],
                        "doc_index": candidate_doc_indices[filtered_indices[j]],
                        "query": q,
                        "text": candidate_texts[filtered_indices[j]],
                        "score": float(filtered_scores[j])
                    })
        return results

    # Parallelize model processing
    with ThreadPoolExecutor() as executor:
        model_results = list(executor.map(process_model, model))
        for results in model_results:
            all_results.extend(results)

    # Fuse results (aggregate chunk scores by document)
    fused_results = fuse_all_results(
        all_results,
        method=fuse_method,
        model_weights=model_weights if fuse_method == "weighted" else None
    )

    # Aggregate by original document
    doc_results = defaultdict(
        lambda: {"scores": [], "text": "", "doc_index": None})
    for result in fused_results:
        doc_idx = result["doc_index"]
        doc_results[doc_idx]["scores"].append(result["score"])
        doc_results[doc_idx]["text"] = texts[doc_idx]  # Store original text
        doc_results[doc_idx]["doc_index"] = doc_idx

    final_results = []
    for doc_idx, data in doc_results.items():
        score = sum(data["scores"]) / \
            len(data["scores"]) if data["scores"] else 0.0
        final_results.append({
            "id": text_ids[doc_idx],
            "rank": None,
            "doc_index": doc_idx,
            "score": score,
            "percent_difference": None,
            "text": data["text"],
            "relevance": None,
            "word_count": len(word_tokenize(data["text"]))
        })

    # Sort and assign ranks
    final_results.sort(key=lambda x: x["score"], reverse=True)
    for idx, result in enumerate(final_results):
        result["rank"] = idx + 1

    # Calculate percent_difference
    if final_results:
        max_score = final_results[0]["score"]
        for result in final_results:
            result["percent_difference"] = round(
                abs(max_score - result["score"]) / max_score * 100, 2
            ) if max_score != 0 else 0.0

    return final_results


def fuse_all_results(
    results: List[Dict[str, Any]],
    method: str = "average",
    model_weights: Optional[List[float]] = None
) -> List[SimilarityResult]:
    """
    Fuses similarity results with advanced methods.

    Args:
        results: List of result dictionaries.
        method: Fusion method ('average', 'max', 'min', 'weighted').
        model_weights: Weights for weighted fusion.

    Returns:
        List of SimilarityResult, sorted by score.
    """
    # Aggregate scores by (id, query, text)
    query_text_data = defaultdict(
        lambda: {"scores": [], "text": None, "doc_index": None})
    for result in results:
        key = (result["id"], result["query"], result["text"])
        query_text_data[key]["scores"].append(result["score"])
        query_text_data[key]["text"] = result["text"]
        query_text_data[key]["doc_index"] = result["doc_index"]

    # Average scores across models
    query_text_averages = {
        key: {
            "text": data["text"],
            "score": float(sum(data["scores"]) / len(data["scores"])),
            "doc_index": data["doc_index"]
        }
        for key, data in query_text_data.items()
    }

    # Fuse query-specific scores for each text
    text_data = defaultdict(
        lambda: {"scores": [], "text": None, "doc_index": None})
    for (id_, query, text), data in query_text_averages.items():
        text_key = (id_, text)
        text_data[text_key]["scores"].append(data["score"])
        text_data[text_key]["text"] = text
        text_data[text_key]["doc_index"] = data["doc_index"]

    # Apply fusion method
    fused_scores = []
    for key, data in text_data.items():
        scores = data["scores"]
        if method == "average":
            score = float(sum(scores) / len(scores))
        elif method == "max":
            score = float(max(scores))
        elif method == "min":
            score = float(min(scores))
        elif method == "weighted":
            normalized_weights = [w / sum(model_weights)
                                  for w in model_weights]
            score = float(
                sum(s * w for s, w in zip(scores, normalized_weights[:len(scores)])))
        fused_scores.append({
            "id": key[0],
            "rank": None,
            "doc_index": data["doc_index"],
            "score": score,
            "percent_difference": None,
            "text": key[1]
        })

    return fused_scores


def write_query(contexts: list[str]) -> str:
    system_prompt = (
        "You are an AI assistant specialized in generating search queries for vector search. "
        "Your task is to create a concise and precise search query in the form of a natural, grammatically correct question. "
        "The question should incorporate specific, descriptive, and contextually relevant keywords from the provided contexts, "
        "capturing their essence while being optimized for vector search. "
        "Output only the question wrapped in a ```text code block, with no additional text."
    )
    # Combine contexts into a single string for the user message
    context = mlx.filter_docs(contexts)
    user_message = f"Contexts:\n{context}"

    response = ""
    for chunk in mlx.stream_chat(
        user_message,
        system_prompt=system_prompt,
        temperature=0.3
    ):
        content = chunk["choices"][0]["message"]["content"]
        response += content
        logger.success(content, flush=True)
    response = extract_block_content(response)
    return response


def rewrite_query(original_query: str) -> str:
    system_prompt = "You are an AI assistant specialized in improving search queries. Your task is to rewrite user queries to be more specific, detailed, and likely to retrieve relevant information."
    response = ""
    for chunk in mlx.stream_chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Rewrite this query: {original_query}"}
        ],
        temperature=0.0
    ):
        content = chunk["choices"][0]["message"]["content"]
        response += content
        logger.success(content, flush=True)
    return response


def rerank_llm(query: str, texts: list[str], top_k: int = 10, threshold: float = 0.0, use_bm25: bool = True, bm25_k: int = 100) -> List[SimilarityResult]:
    # query = write_query(texts)

    logger.info(
        f"Reranking {len(texts)} for query: {query}")
    results = query_similarity_scores(
        query=query,
        texts=texts,
        # ids=article_ids,
        top_k=top_k,
        threshold=threshold,
        use_bm25=use_bm25,
        bm25_k=bm25_k,
        # model=["all-MiniLM-L12-v2", "distilbert-base-nli-stsb-mean-tokens"],
        model=["all-MiniLM-L12-v2"],
        fuse_method="average",
        metrics="cosine",
        domain=None,
        use_index=len(texts) > 1000,
    )
    return results


def news_article_search(
    query: str,
    articles: List[str],
    article_ids: List[str],
    threshold: float = 0.4,
    top_k: int = 5,
    use_bm25: bool = True
) -> List[SimilarityResult]:
    """
    Reranks web-scraped news articles based on a query.

    Args:
        query: User query (e.g., "AI in healthcare").
        articles: List of scraped article texts.
        article_ids: Unique IDs for articles.
        threshold: Minimum similarity score.
        top_k: Number of top articles to return.
        use_bm25: Whether to use BM25 for candidate selection.

    Returns:
        List of reranked SimilarityResult objects.
    """
    logger.info(f"Reranking {len(articles)} news articles for query: {query}")
    results = query_similarity_scores(
        query=query,
        texts=articles,
        ids=article_ids,
        threshold=threshold,
        model=["all-MiniLM-L6-v2"],
        fuse_method="average",
        metrics="cosine",
        domain="news",
        use_index=len(articles) > 1000,
        top_k=top_k,
        use_bm25=use_bm25,
        bm25_k=100
    )
    return results


def tech_blog_recommendation(
    queries: List[str],
    posts: List[str],
    post_ids: List[str],
    threshold: float = 0.3,
    top_k: int = 5,
    use_bm25: bool = False
) -> List[SimilarityResult]:
    """
    Recommends web-scraped tech blog posts based on multiple interest queries.

    Args:
        queries: List of user interest queries (e.g., ["machine learning", "AI trends"]).
        posts: List of scraped blog post texts.
        post_ids: Unique IDs for posts.
        threshold: Minimum similarity score.
        top_k: Number of top posts to return.
        use_bm25: Whether to use BM25 for candidate selection.

    Returns:
        List of reranked SimilarityResult objects.
    """
    logger.info(
        f"Recommending {len(posts)} blog posts for {len(queries)} queries")
    results = query_similarity_scores(
        query=queries,
        texts=posts,
        ids=post_ids,
        threshold=threshold,
        model=["all-MiniLM-L6-v2", "distilbert-base-nli-stsb-mean-tokens"],
        fuse_method="weighted",
        model_weights=[0.6, 0.4],
        metrics="cosine",
        domain="blog",
        use_index=len(posts) > 1000,
        top_k=top_k,
        use_bm25=use_bm25,
        bm25_k=100
    )
    return results


def ecommerce_product_scraping(
    query: str,
    product_listings: List[str],
    listing_ids: List[str],
    threshold: float = 0.5,
    top_k: int = 5,
    use_bm25: bool = True
) -> List[SimilarityResult]:
    """
    Reranks scraped e-commerce product listings based on a query.

    Args:
        query: Product search query (e.g., "wireless earbuds").
        product_listings: List of scraped product listing texts.
        listing_ids: Unique IDs for listings.
        threshold: Minimum similarity score.
        top_k: Number of top listings to return.
        use_bm25: Whether to use BM25 for candidate selection.

    Returns:
        List of reranked SimilarityResult objects.
    """
    logger.info(
        f"Reranking {len(product_listings)} product listings for query: {query}")
    results = query_similarity_scores(
        query=query,
        texts=product_listings,
        ids=listing_ids,
        threshold=threshold,
        model=["all-MiniLM-L6-v2"],
        fuse_method="average",
        metrics="cosine",
        domain="ecommerce",
        use_index=len(product_listings) > 500,
        top_k=top_k,
        use_bm25=use_bm25,
        bm25_k=100
    )
    return results


def forum_thread_prioritization(
    query: str,
    threads: List[str],
    thread_ids: List[str],
    threshold: float = 0.4,
    top_k: int = 5,
    use_bm25: bool = True
) -> List[SimilarityResult]:
    """
    Prioritizes web-scraped forum threads based on a query.

    Args:
        query: Technical query (e.g., "Python error handling").
        threads: List of scraped forum thread texts.
        thread_ids: Unique IDs for threads.
        threshold: Minimum similarity score.
        top_k: Number of top threads to return.
        use_bm25: Whether to use BM25 for candidate selection.

    Returns:
        List of reranked SimilarityResult objects.
    """
    logger.info(
        f"Prioritizing {len(threads)} forum threads for query: {query}")
    results = query_similarity_scores(
        query=query,
        texts=threads,
        ids=thread_ids,
        threshold=threshold,
        model=["all-MiniLM-L6-v2"],
        fuse_method="average",
        metrics="cosine",
        domain="forum",
        use_index=len(threads) > 1000,
        top_k=top_k,
        use_bm25=use_bm25,
        bm25_k=100
    )
    return results
