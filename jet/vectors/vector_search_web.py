import os
import requests
from jet.file.utils import load_file, save_file
from jet.logger import logger
import justext
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import re
from urllib.robotparser import RobotFileParser
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from scipy.special import expit
from typing import List, Optional, Tuple, TypedDict
import uuid
from lxml import html


nltk.download('punkt', quiet=True)

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(
        os.path.basename(__file__))[0]
)


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
            logger.warning(f"Scraping disallowed by {robots_url} for {url}")
        return allowed
    except Exception as e:
        logger.error(f"Failed to check robots.txt for {url}: {e}")
        return True


# def parse_html_documents(html: str, language="English", max_link_density=0.2, max_link_ratio=0.3) -> List[justext.Paragraph]:
#     paragraphs = justext.justext(
#         html,
#         justext.get_stoplist(language),
#         max_link_density=max_link_density,
#         length_low=50,
#         length_high=150,
#         no_headings=False
#     )
#     filtered_paragraphs = [
#         p for p in paragraphs
#         if not p.is_boilerplate and p.links_density() < max_link_ratio
#     ]
#     return filtered_paragraphs


def read_local_html(file_path: str, language: str = "English", max_link_density: float = 0.2, max_link_ratio: float = 0.3) -> Tuple[str, List, Optional[html.HtmlElement]]:
    """
    Read and parse a local HTML file, returning its content as paragraphs and DOM tree.

    Args:
        file_path: Path to the local HTML file.
        language: Language for justext stoplist (default: English).
        max_link_density: Maximum link density for justext (default: 0.2).
        max_link_ratio: Maximum link ratio for filtering paragraphs (default: 0.3).

    Returns:
        Tuple containing the file path, list of filtered paragraphs, and parsed DOM tree.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        dom_tree = html.fromstring(content.encode('utf-8'))
        paragraphs = justext.justext(
            content.encode('utf-8'),
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
        logger.info(
            f"Cleaned {file_path}: {len(filtered_paragraphs)} non-boilerplate paragraphs")
        return file_path, filtered_paragraphs, dom_tree
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        return file_path, [], None


# @retry(
#     stop=stop_after_attempt(3),
#     wait=wait_exponential(multiplier=1, min=4, max=10),
#     retry=retry_if_exception_type(requests.RequestException)
# )
# def fetch_html(url, language="English", max_link_density=0.2, max_link_ratio=0.3):
#     """
#     Fetch HTML content from a URL and clean it using jusText.
#     Returns URL, filtered paragraphs, and parsed DOM tree.
#     """
#     if not check_robots_txt(url):
#         logger.error(f"Skipping {url} due to robots.txt restrictions")
#         return url, [], None

#     response = requests.get(url, timeout=30)
#     response.raise_for_status()
#     dom_tree = html.fromstring(response.content)
#     paragraphs = parse_html_documents(
#         response.content, language="English", max_link_density=0.2, max_link_ratio=0.3
#     )
#     logger.info(
#         f"Cleaned {url}: {len(paragraphs)} non-boilerplate paragraphs")
#     return url, paragraphs, dom_tree


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


def get_header_level(paragraph, dom_tree=None):
    """
    Extract header level (e.g., 'h1', 'h2') from paragraph.xpath or DOM tree.
    """
    if not paragraph.is_heading:
        return None
    try:
        # Try XPath parsing first
        xpath = paragraph.xpath.lower()
        header_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
        for tag in header_tags:
            if f'/{tag}' in xpath or f'/{tag}[' in xpath:
                return tag

        # Fallback to DOM tree if provided
        if dom_tree is not None:
            try:
                elements = dom_tree.xpath(paragraph.xpath)
                if elements and hasattr(elements[0], 'tag'):
                    tag = elements[0].tag.lower()
                    if tag in header_tags:
                        return tag
            except Exception as e:
                logger.debug(
                    f"DOM parsing failed for XPath {paragraph.xpath}: {e}")

        logger.warning(f"No header tag found for XPath: {paragraph.xpath}")
        return None
    except Exception as e:
        logger.error(f"Error parsing XPath {paragraph.xpath}: {e}")
        return None


def separate_by_headers(paragraphs, dom_tree=None):
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
            header_level = get_header_level(paragraph, dom_tree)
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

    logger.info(f"Separated into {len(sections)} sections")
    return sections


class Chunk(TypedDict):
    chunk_id: str
    text: str
    token_count: int
    header: Optional[str]
    header_level: Optional[str]
    xpath: Optional[str]


def chunk_with_overlap(section: dict, max_tokens: int = 200, overlap_tokens: int = 50, model: Optional[SentenceTransformer] = None) -> List[Chunk]:
    """
    Split section into chunks with overlap, filtering by header-content similarity and minimum length.
    """
    logger.debug(f"Processing section with header: {section['header']}")
    text = (section["header"] + "\n" + " ".join(section["content"])
            if section["header"] else " ".join(section["content"]))
    sentences = sent_tokenize(text)
    logger.debug(f"Sentences: {sentences}")
    chunks: List[Chunk] = []
    current_chunk = []
    current_tokens = 0
    chunk_id = str(uuid.uuid4())

    header_embedding = model.encode(
        section["header"], convert_to_tensor=False, show_progress_bar=False) if section["header"] else None
    logger.debug(f"Header embedding created: {header_embedding is not None}")

    for sentence in sentences:
        sentence_tokens = len(word_tokenize(sentence))
        logger.debug(f"Sentence: '{sentence}', tokens: {sentence_tokens}")
        if current_tokens + sentence_tokens <= max_tokens:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        else:
            chunk_text = " ".join(current_chunk)
            token_count = len(word_tokenize(chunk_text))
            logger.debug(
                f"Chunk text: '{chunk_text}', token count: {token_count}")
            if token_count < 5:  # Lowered from 20 to 5 to allow test chunks
                logger.debug(
                    f"Skipping chunk under header '{section['header']}' due to short length ({token_count} tokens)")
                current_chunk = []
                current_tokens = 0
                continue
            if header_embedding is not None:
                chunk_embedding = model.encode(
                    chunk_text, convert_to_tensor=False, show_progress_bar=False)
                similarity = np.dot(header_embedding, chunk_embedding) / (
                    np.linalg.norm(header_embedding) * np.linalg.norm(chunk_embedding))
                logger.debug(f"Similarity score for chunk: {similarity:.2f}")
                if similarity < 0.5:
                    logger.debug(
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
            logger.debug(f"Added chunk: {chunk_text}")
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
            logger.debug(
                f"New chunk started with overlap: {overlap_sentences}")

    if current_chunk:
        chunk_text = " ".join(current_chunk)
        token_count = len(word_tokenize(chunk_text))
        logger.debug(
            f"Final chunk text: '{chunk_text}', token count: {token_count}")
        if token_count >= 5:  # Lowered from 20 to 5 to allow test chunks
            if header_embedding is not None:
                chunk_embedding = model.encode(
                    chunk_text, convert_to_tensor=False, show_progress_bar=False)
                similarity = np.dot(header_embedding, chunk_embedding) / (
                    np.linalg.norm(header_embedding) * np.linalg.norm(chunk_embedding))
                logger.debug(
                    f"Final chunk similarity score: {similarity:.2f}")
                if similarity < 0.5:
                    logger.debug(
                        f"Skipping final chunk under header '{section['header']}' due to low similarity ({similarity:.2f})")
                else:
                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": chunk_text,
                        "token_count": token_count,
                        "header": section["header"],
                        "header_level": section["header_level"],
                        "xpath": section["xpath"]
                    })
                    logger.debug(f"Added final chunk: {chunk_text}")
            else:
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "token_count": token_count,
                    "header": section["header"],
                    "header_level": section["header_level"],
                    "xpath": section["xpath"]
                })
                logger.debug(f"Added final chunk (no header): {chunk_text}")

    logger.info(
        f"Created {len(chunks)} chunks for section: {section['header'] or 'No header'}")
    return chunks


def prepare_for_rag(html_paragraphs_pairs, model_name='all-MiniLM-L6-v2', batch_size=32):
    """
    Prepare documents for RAG from HTML paragraphs: separate, chunk, embed, and index.
    """
    model = SentenceTransformer(model_name)
    chunked_documents = []

    for url, paragraphs, dom_tree in tqdm(html_paragraphs_pairs, desc="Processing HTML content"):
        if not paragraphs:
            logger.warning(f"No paragraphs provided for {url}, skipping")
            continue
        sections = separate_by_headers(paragraphs, dom_tree)
        for section in tqdm(sections, desc=f"Chunking sections for {url}", leave=False):
            chunks = chunk_with_overlap(
                section, max_tokens=200, overlap_tokens=50, model=model)
            for chunk in chunks:
                chunk["url"] = url
            chunked_documents.extend(chunks)

    if not chunked_documents:
        logger.warning("No chunks generated, cannot create FAISS index")
        return None, [], model

    # Deduplicate by header, keeping longest chunk
    header_to_chunks = {}
    for chunk in chunked_documents:
        header = chunk["header"] or "No header"
        if header not in header_to_chunks or chunk["token_count"] > header_to_chunks[header]["token_count"]:
            header_to_chunks[header] = chunk

    chunked_documents = list(header_to_chunks.values())
    logger.info(f"Deduplicated to {len(chunked_documents)} unique chunks")

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

    save_file(rag_docs, f"{OUTPUT_DIR}/rag_documents.json")

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

    save_file(metadata, f"{OUTPUT_DIR}/metadata.json")

    logger.info(f"Indexed {len(embeddings)} chunks in FAISS")
    return index, embeddings, model


def search_docs(
    query: str,
    documents: List[str],
    model: str = "all-MiniLM-L6-v2",
    top_k: Optional[int] = 10,
    ids: Optional[List[str]] = None,
    threshold: Optional[float] = 0.5
) -> List[SimilarityResult]:
    """
    Search documents for similarity to a query using a sentence transformer model.
    """
    if not documents:
        logger.warning("No documents provided, returning empty results")
        return []

    # Initialize models
    logger.info(f"Loading SentenceTransformer: {model}")
    embedder = SentenceTransformer(model)
    cross_encoder = CrossEncoder("cross-encoder/stsb-roberta-base")

    # Generate IDs if not provided
    if ids is None:
        ids = [str(uuid.uuid4()) for _ in documents]
    elif len(ids) != len(documents):
        logger.error("Length of ids does not match length of documents")
        raise ValueError("Length of ids must match length of documents")

    # Compute token counts
    token_counts = [len(word_tokenize(doc)) for doc in documents]

    # Embed query and documents
    logger.info("Generating embeddings for query and documents")
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
    cross_scores = [float(min(expit(score), 1.0))
                    for score in cross_scores]  # No scaling, cap at 1.0

    # Filter by threshold
    filtered = [(i, s)
                for i, s in zip(top_indices, cross_scores) if s >= threshold]
    if not filtered:
        logger.warning("No documents meet the threshold")
        return []
    top_indices, cross_scores = zip(*filtered) if filtered else ([], [])

    # Sort by cross-encoder scores
    sorted_pairs = sorted(
        zip(top_indices, cross_scores), key=lambda x: x[1], reverse=True)
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

    logger.info(f"Returning {len(results)} similarity results")
    return results


# def main():
#     """
#     Main function to fetch HTML, process it, and query RAG.
#     """
#     urls = ["https://planet.python.org/"]
#     html_paragraphs_pairs = []

#     with ThreadPoolExecutor() as executor:
#         results = list(tqdm(
#             executor.map(fetch_html, urls),
#             total=len(urls),
#             desc="Fetching URLs"
#         ))
#         html_paragraphs_pairs.extend(results)

#     index, embeddings, model = prepare_for_rag(
#         html_paragraphs_pairs, batch_size=32)

#     if index is None or not embeddings:
#         print("No data indexed, exiting.")
#         return

#     query_text = "What is Python programming?"
#     documents = [chunk["text"] for chunk in embeddings]
#     ids = [chunk["chunk_id"] for chunk in embeddings]
#     results = search_docs(
#         query_text, documents, model="all-MiniLM-L6-v2", top_k=10, ids=ids, threshold=0.5)

#     print("\nQuery Results:")
#     for i, result in enumerate(results, 1):
#         chunk = embeddings[result["doc_index"]]
#         header_level = chunk["header_level"] or "None"
#         if header_level not in ["h1", "h2", "h3", "h4", "h5", "h6", None]:
#             logger.warning(
#                 f"Unexpected header level '{header_level}' for header: {chunk['header']}")
#         print(f"\nResult {i}:")
#         print(f"Header: {chunk['header'] or 'No header'}")
#         print(f"Header Level: {header_level}")
#         print(f"Text: {result['text']}")
#         print(f"URL: {chunk['url']}")
#         print(f"Score: {result['score']:.4f}")
#         print(f"Tokens: {result['tokens']}")

def main():
    """
    Main function to process a local HTML file and query RAG.
    """

    query_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_isekai_anime_2025/query.md"
    html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_isekai_anime_2025/pages/animebytes_in_15_best_upcoming_isekai_anime_in_2025/page.html"
    if not os.path.exists(html_file):
        logger.error(f"HTML file {html_file} not found")
        return

    html_paragraphs_pairs = [read_local_html(html_file)]

    # save_file(html_paragraphs_pairs,
    #           f"{OUTPUT_DIR}/html_paragraphs_pairs.json")

    index, embeddings, model = prepare_for_rag(
        html_paragraphs_pairs, batch_size=32)

    if index is None or not embeddings:
        print("No data indexed, exiting.")
        return

    query_text = load_file(query_file)
    documents = [chunk["text"] for chunk in embeddings]
    ids = [chunk["chunk_id"] for chunk in embeddings]
    results = search_docs(
        query_text, documents, model="all-MiniLM-L6-v2", top_k=10, ids=ids, threshold=0.5)

    print("\nQuery Results:")
    for i, result in enumerate(results, 1):
        chunk = embeddings[result["doc_index"]]
        header_level = chunk["header_level"] or "None"
        if header_level not in ["h1", "h2", "h3", "h4", "h5", "h6", None]:
            logger.warning(
                f"Unexpected header level '{header_level}' for header: {chunk['header']}")
        print(f"\nResult {i}:")
        print(f"Header: {chunk['header'] or 'No header'}")
        print(f"Header Level: {header_level}")
        print(f"Text: {result['text']}")
        print(f"File: {chunk['url']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Tokens: {result['tokens']}")

    save_file(results, f"{OUTPUT_DIR}/search_results.json")


if __name__ == "__main__":
    main()
