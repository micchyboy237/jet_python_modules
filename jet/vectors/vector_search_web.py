import requests
import justext
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
import json
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import re
from urllib.robotparser import RobotFileParser
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from scipy.special import expit
from typing import List, Optional, TypedDict
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
nltk.download('punkt', quiet=True)


class SimilarityResult(TypedDict):
    """
    Represents a single similarity result for a text.
    """
    id: str
    rank: int
    doc_index: int
    score: float
    text: str
    tokens: int


def check_robots_txt(url):
    """
    Check if scraping is allowed by robots.txt.
    """
    try:
        rp = RobotFileParser()
        robots_url = f"{url.rsplit('/', 1)[0]}/robots.txt"
        rp.set_url(robots_url)
        rp.read()
        user_agent = "*"
        allowed = rp.can_fetch(user_agent, url)
        if not allowed:
            logging.warning(f"Scraping disallowed by {robots_url} for {url}")
        return allowed
    except Exception as e:
        logging.error(f"Failed to check robots.txt for {url}: {e}")
        return True


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(requests.RequestException)
)
def fetch_html(url, language="English", max_link_density=0.2, max_link_ratio=0.3):
    """
    Fetch HTML content from a URL and clean it using jusText.
    Returns URL and filtered paragraphs.
    """
    if not check_robots_txt(url):
        logging.error(f"Skipping {url} due to robots.txt restrictions")
        return url, []

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    paragraphs = justext.justext(
        response.content,
        justext.get_stoplist(language),
        max_link_density=max_link_density,
        length_low=50,
        length_high=150,
        no_headings=False
    )
    filtered_paragraphs = [
        p for p in paragraphs
        if not p.is_boilerplate and p.links_density() < max_link_ratio
    ]
    logging.info(
        f"Cleaned {url}: {len(filtered_paragraphs)} non-boilerplate paragraphs")
    return url, filtered_paragraphs


def is_valid_header(header):
    """
    Filter out generic or date-based headers.
    """
    if not header:
        return True
    generic_keywords = {'planet', 'articles', 'tutorials', 'jobs', 'topic',
                        'further', 'why', 'looking', 'wrapping', 'learn', 'system', 'configuration'}
    date_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b'
    if any(keyword in header.lower() for keyword in generic_keywords) or re.match(date_pattern, header):
        return False
    return True


def get_header_level(paragraph):
    """
    Extract header level (e.g., 'h1', 'h2') from paragraph.xpath.
    """
    if not paragraph.is_heading:
        return None
    try:
        # XPath example: /html/body/div[1]/h2[1]
        xpath = paragraph.xpath.lower()
        # Find last tag in XPath that is a header (h1-h6)
        header_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
        for tag in header_tags:
            if f'/{tag}' in xpath or f'/{tag}[' in xpath:
                return tag
        logging.debug(f"No header tag found in XPath: {xpath}")
        return None
    except Exception as e:
        logging.error(f"Error parsing XPath {paragraph.xpath}: {e}")
        return None


def separate_by_headers(paragraphs):
    """
    Group paragraphs into sections based on headers (h1-h6) and identify header level.
    """
    sections = []
    current_section = {"header": None,
                       "header_level": None, "content": [], "xpath": None}
    header_tags = {"h1", "h2", "h3", "h4", "h5", "h6"}

    for paragraph in paragraphs:
        if paragraph.is_heading:
            if (current_section["content"] or current_section["header"]) and is_valid_header(current_section["header"]):
                sections.append(current_section)
            header_level = get_header_level(paragraph)
            current_section = {
                "header": paragraph.text,
                "header_level": header_level,
                "content": [],
                "xpath": paragraph.xpath
            }
        else:
            current_section["content"].append(paragraph.text)

    if (current_section["content"] or current_section["header"]) and is_valid_header(current_section["header"]):
        sections.append(current_section)

    logging.info(f"Separated into {len(sections)} sections")
    return sections


def chunk_with_overlap(section, max_tokens=200, overlap_tokens=50, model=None):
    """
    Split section into chunks with overlap, filtering by header-content similarity and minimum length.
    """
    text = (section["header"] + "\n" + " ".join(section["content"])
            if section["header"] else " ".join(section["content"]))
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_tokens = 0
    chunk_id = str(uuid.uuid4())

    header_embedding = model.encode(
        section["header"], convert_to_tensor=False, show_progress_bar=False) if section["header"] else None

    for sentence in sentences:
        sentence_tokens = len(word_tokenize(sentence))
        if current_tokens + sentence_tokens <= max_tokens:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        else:
            chunk_text = " ".join(current_chunk)
            token_count = len(word_tokenize(chunk_text))
            if token_count < 20:
                logging.debug(
                    f"Skipping chunk under header '{section['header']}' due to short length ({token_count} tokens)")
                current_chunk = []
                current_tokens = 0
                continue
            if header_embedding is not None:
                chunk_embedding = model.encode(
                    chunk_text, convert_to_tensor=False, show_progress_bar=False)
                similarity = np.dot(header_embedding, chunk_embedding) / (
                    np.linalg.norm(header_embedding) * np.linalg.norm(chunk_embedding))
                if similarity < 0.5:
                    logging.debug(
                        f"Skipping chunk under header '{section['header']}' due to low similarity ({similarity:.2f})")
                    current_chunk = []
                    current_tokens = 0
                    continue
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "token_count": token_count,
                "header": section["header"],
                "header_level": section["header_level"],
                "xpath": section["xpath"]
            })
            overlap_sentences = []
            overlap_count = 0
            for s in current_chunk[::-1]:
                s_tokens = len(word_tokenize(s))
                if overlap_count + s_tokens <= overlap_tokens:
                    overlap_sentences.insert(0, s)
                    overlap_count += s_tokens
                else:
                    break
            current_chunk = overlap_sentences + [sentence]
            current_tokens = overlap_count + sentence_tokens
            chunk_id = str(uuid.uuid4())

    if current_chunk:
        chunk_text = " ".join(current_chunk)
        token_count = len(word_tokenize(chunk_text))
        if token_count >= 20:
            if header_embedding is not None:
                chunk_embedding = model.encode(
                    chunk_text, convert_to_tensor=False, show_progress_bar=False)
                similarity = np.dot(header_embedding, chunk_embedding) / (
                    np.linalg.norm(header_embedding) * np.linalg.norm(chunk_embedding))
                if similarity < 0.5:
                    logging.debug(
                        f"Skipping chunk under header '{section['header']}' due to low similarity ({similarity:.2f})")
                else:
                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": chunk_text,
                        "token_count": token_count,
                        "header": section["header"],
                        "header_level": section["header_level"],
                        "xpath": section["xpath"]
                    })
            else:
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "token_count": token_count,
                    "header": section["header"],
                    "header_level": section["header_level"],
                    "xpath": section["xpath"]
                })

    logging.info(
        f"Created {len(chunks)} chunks for section: {section['header'] or 'No header'}")
    return chunks


def prepare_for_rag(html_paragraphs_pairs, model_name='all-MiniLM-L6-v2', batch_size=32):
    """
    Prepare documents for RAG from HTML paragraphs: separate, chunk, embed, and index.
    """
    model = SentenceTransformer(model_name)
    chunked_documents = []

    for url, paragraphs in tqdm(html_paragraphs_pairs, desc="Processing HTML content"):
        if not paragraphs:
            logging.warning(f"No paragraphs provided for {url}, skipping")
            continue
        sections = separate_by_headers(paragraphs)
        for section in tqdm(sections, desc=f"Chunking sections for {url}", leave=False):
            chunks = chunk_with_overlap(
                section, max_tokens=200, overlap_tokens=50, model=model)
            for chunk in chunks:
                chunk["url"] = url
            chunked_documents.extend(chunks)

    if not chunked_documents:
        logging.warning("No chunks generated, cannot create FAISS index")
        return None, [], model

    # Deduplicate by header, keeping longest chunk
    header_to_chunks = {}
    for chunk in chunked_documents:
        header = chunk["header"] or "No header"
        if header not in header_to_chunks or chunk["token_count"] > header_to_chunks[header]["token_count"]:
            header_to_chunks[header] = chunk

    chunked_documents = list(header_to_chunks.values())
    logging.info(f"Deduplicated to {len(chunked_documents)} unique chunks")

    # Save RAG documents to file
    rag_docs = [
        {
            "chunk_id": doc["chunk_id"],
            "url": doc["url"],
            "header": doc["header"],
            "header_level": doc["header_level"],
            "text": doc["text"],
            "token_count": doc["token_count"],
            "xpath": doc["xpath"]
        } for doc in chunked_documents
    ]
    with open("rag_documents.json", "w") as f:
        json.dump(rag_docs, f, indent=2)
    logging.info(f"Saved {len(rag_docs)} RAG documents to rag_documents.json")

    # Generate embeddings in batches
    embeddings = []
    texts = [chunk["text"] for chunk in chunked_documents]
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = model.encode(
            batch_texts, convert_to_tensor=False, show_progress_bar=False, normalize_embeddings=True)
        for j, embedding in enumerate(batch_embeddings):
            chunked_documents[i + j]["embedding"] = embedding
            embeddings.append(chunked_documents[i + j])

    # Create FAISS index with cosine similarity
    embedding_matrix = np.array([doc["embedding"]
                                for doc in embeddings]).astype('float32')
    index = faiss.IndexFlatIP(embedding_matrix.shape[1])
    index.add(embedding_matrix)

    # Save metadata
    metadata = [
        {
            "chunk_id": doc["chunk_id"],
            "url": doc["url"],
            "header": doc["header"],
            "header_level": doc["header_level"],
            "text": doc["text"],
            "token_count": doc["token_count"],
            "xpath": doc["xpath"],
            "index": i
        } for i, doc in enumerate(embeddings)
    ]
    with open("metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logging.info(f"Indexed {len(embeddings)} chunks in FAISS")
    return index, embeddings, model


def search_docs(
    query: str,
    documents: List[str],
    model: str = "all-MiniLM-L6-v2",
    top_k: Optional[int] = 10,
    ids: Optional[List[str]] = None,
    threshold: Optional[float] = None
) -> List[SimilarityResult]:
    """
    Search documents for similarity to a query using a sentence transformer model.
    """
    if not documents:
        logging.warning("No documents provided, returning empty results")
        return []

    # Initialize models
    logging.info(f"Loading SentenceTransformer: {model}")
    embedder = SentenceTransformer(model)
    cross_encoder = CrossEncoder("cross-encoder/stsb-roberta-base")

    # Generate IDs if not provided
    if ids is None:
        ids = [str(uuid.uuid4()) for _ in documents]
    elif len(ids) != len(documents):
        logging.error("Length of ids does not match length of documents")
        raise ValueError("Length of ids must match length of documents")

    # Compute token counts
    token_counts = [len(word_tokenize(doc)) for doc in documents]

    # Embed query and documents
    logging.info("Generating embeddings for query and documents")
    query_embedding = embedder.encode(
        query, convert_to_tensor=False, show_progress_bar=False, normalize_embeddings=True)
    doc_embeddings = embedder.encode(
        documents, convert_to_tensor=False, show_progress_bar=False, normalize_embeddings=True)

    # Compute cosine similarities
    similarities = np.dot(doc_embeddings, query_embedding) / (
        np.linalg.norm(doc_embeddings, axis=1) *
        np.linalg.norm(query_embedding)
    )

    # Get top-k indices
    top_k = min(top_k, len(documents))
    top_indices = np.argsort(similarities)[::-1][:top_k]

    # Re-rank with cross-encoder
    pairs = [[query, documents[idx]] for idx in top_indices]
    cross_scores = cross_encoder.predict(pairs, show_progress_bar=False)
    cross_scores = [float(min(expit(score) * 1.5, 1.0))
                    for score in cross_scores]

    # Filter by threshold
    if threshold is not None:
        filtered = [(i, s) for i, s in zip(
            top_indices, cross_scores) if s >= threshold]
        if not filtered:
            logging.warning("No documents meet the threshold")
            return []
        top_indices, cross_scores = zip(*filtered) if filtered else ([], [])

    # Filter by keywords (Python or programming)
    filtered_indices = []
    filtered_scores = []
    for i, idx in enumerate(top_indices):
        text = documents[idx].lower()
        if "python" in text or "programming" in text:
            filtered_indices.append(idx)
            filtered_scores.append(cross_scores[i])

    if not filtered_indices:
        logging.warning("No documents pass keyword filter")
        return []

    # Sort by cross-encoder scores
    sorted_pairs = sorted(
        zip(filtered_indices, filtered_scores), key=lambda x: x[1], reverse=True)
    filtered_indices, cross_scores = zip(
        *sorted_pairs) if sorted_pairs else ([], [])

    # Create results
    results = []
    for rank, (idx, score) in enumerate(zip(filtered_indices[:top_k], cross_scores[:top_k]), 1):
        results.append({
            "id": ids[idx],
            "rank": rank,
            "doc_index": int(idx),
            "score": score,
            "text": documents[idx][:200] + "..." if len(documents[idx]) > 200 else documents[idx],
            "tokens": token_counts[idx]
        })

    logging.info(f"Returning {len(results)} similarity results")
    return results


def main():
    """
    Main function to fetch HTML, process it, and query RAG.
    """
    urls = ["https://planet.python.org/"]
    html_paragraphs_pairs = []

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(fetch_html, urls),
            total=len(urls),
            desc="Fetching URLs"
        ))
        html_paragraphs_pairs.extend(results)

    index, embeddings, model = prepare_for_rag(
        html_paragraphs_pairs, batch_size=32)

    if index is None or not embeddings:
        print("No data indexed, exiting.")
        return

    query_text = "What is Python programming?"
    documents = [chunk["text"] for chunk in embeddings]
    ids = [chunk["chunk_id"] for chunk in embeddings]
    results = search_docs(
        query_text, documents, model="all-MiniLM-L6-v2", top_k=10, ids=ids, threshold=0.4)

    print("\nQuery Results:")
    for i, result in enumerate(results, 1):
        chunk = embeddings[result["doc_index"]]
        print(f"\nResult {i}:")
        print(f"Header: {chunk['header'] or 'No header'}")
        print(f"Header Level: {chunk['header_level'] or 'None'}")
        print(f"Text: {result['text']}")
        print(f"URL: {chunk['url']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Tokens: {result['tokens']}")


if __name__ == "__main__":
    main()
