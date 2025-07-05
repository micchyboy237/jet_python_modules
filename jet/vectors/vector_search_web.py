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
from scipy.special import expit  # For sigmoid

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
nltk.download('punkt', quiet=True)


def check_robots_txt(url):
    """
    Check if scraping is allowed by robots.txt.
    Returns True if allowed, False otherwise.
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
def clean_html(url, language="English", max_link_density=0.2, max_link_ratio=0.3):
    """
    Clean HTML using jusText, filtering out boilerplate and high link-to-text ratio content.
    """
    if not check_robots_txt(url):
        logging.error(f"Skipping {url} due to robots.txt restrictions")
        return []

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
    return filtered_paragraphs


def is_valid_header(header):
    """
    Filter out generic or date-based headers.
    """
    if not header:
        return True
    generic_keywords = {'planet', 'articles', 'tutorials', 'jobs', 'topic'}
    date_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b'
    if any(keyword in header.lower() for keyword in generic_keywords) or re.match(date_pattern, header):
        return False
    return True


def separate_by_headers(paragraphs):
    """
    Group paragraphs into sections based on headers (h1-h6).
    Returns a list of sections with header, content, and xpath.
    """
    sections = []
    current_section = {"header": None, "content": [], "xpath": None}
    header_tags = {"h1", "h2", "h3", "h4", "h5", "h6"}

    for paragraph in paragraphs:
        if paragraph.is_heading:
            if (current_section["content"] or current_section["header"]) and is_valid_header(current_section["header"]):
                sections.append(current_section)
            current_section = {
                "header": paragraph.text,
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
    Split section into chunks with overlap, filtering by header-content similarity.
    """
    text = (section["header"] + "\n" + " ".join(section["content"])
            if section["header"] else " ".join(section["content"]))
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_tokens = 0
    chunk_id = 0

    # Compute header embedding if available
    header_embedding = model.encode(
        section["header"], convert_to_tensor=False, show_progress_bar=False) if section["header"] else None

    for sentence in sentences:
        sentence_tokens = len(word_tokenize(sentence))
        if current_tokens + sentence_tokens <= max_tokens:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        else:
            chunk_text = " ".join(current_chunk)
            # Filter by header-content similarity
            if header_embedding is not None:
                chunk_embedding = model.encode(
                    chunk_text, convert_to_tensor=False, show_progress_bar=False)
                similarity = np.dot(header_embedding, chunk_embedding) / (
                    np.linalg.norm(header_embedding) * np.linalg.norm(chunk_embedding))
                if similarity < 0.5:  # Threshold for relevance
                    logging.debug(
                        f"Skipping chunk under header '{section['header']}' due to low similarity ({similarity:.2f})")
                    current_chunk = []
                    current_tokens = 0
                    continue
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "token_count": current_tokens,
                "header": section["header"],
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
            chunk_id += 1

    if current_chunk:
        chunk_text = " ".join(current_chunk)
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
                    "token_count": current_tokens,
                    "header": section["header"],
                    "xpath": section["xpath"]
                })
        else:
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "token_count": current_tokens,
                "header": section["header"],
                "xpath": section["xpath"]
            })

    logging.info(
        f"Created {len(chunks)} chunks for section: {section['header'] or 'No header'}")
    return chunks


def prepare_for_rag(urls, model_name='all-MiniLM-L6-v2', batch_size=32):
    """
    Prepare documents for RAG: clean, separate, chunk, embed, and index.
    """
    model = SentenceTransformer(model_name)
    chunked_documents = []

    # Process URLs in parallel with progress bar
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(lambda url: clean_html(
                url, max_link_density=0.15, max_link_ratio=0.3), urls),
            total=len(urls),
            desc="Processing URLs"
        ))

    for url, paragraphs in zip(urls, results):
        if not paragraphs:
            logging.warning(f"No paragraphs extracted for {url}, skipping")
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

    # Deduplicate chunks by header and content
    unique_chunks = []
    seen = set()
    for chunk in chunked_documents:
        key = (chunk["header"], chunk["text"])
        if key not in seen:
            seen.add(key)
            unique_chunks.append(chunk)

    chunked_documents = unique_chunks
    logging.info(f"Deduplicated to {len(chunked_documents)} unique chunks")

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
            "text": doc["text"],
            "xpath": doc["xpath"],
            "index": i
        } for i, doc in enumerate(embeddings)
    ]
    with open("metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logging.info(f"Indexed {len(embeddings)} chunks in FAISS")
    return index, embeddings, model


def query_rag(index, embeddings, model, query_text, k=20, cross_encoder_model='cross-encoder/ms-marco-MiniLM-L-12-v2'):
    """
    Query the RAG system and return top-10 results sorted by cross-encoder score.
    """
    if index is None or not embeddings:
        logging.error("No index or embeddings available for querying")
        return []

    cross_encoder = CrossEncoder(cross_encoder_model)
    query_embedding = model.encode(query_text, convert_to_tensor=False,
                                   show_progress_bar=False, normalize_embeddings=True).astype('float32')
    D, I = index.search(np.array([query_embedding]), k)
    results = []

    # Prepare pairs for cross-encoder re-ranking
    pairs = [[query_text, embeddings[idx]["text"]] for idx in I[0]]
    cross_scores = cross_encoder.predict(pairs)

    for idx, cross_score in zip(I[0], cross_scores):
        # Apply sigmoid to normalize scores to [0, 1]
        normalized_score = expit(cross_score)
        results.append({
            "header": embeddings[idx]["header"],
            "text": embeddings[idx]["text"],
            "url": embeddings[idx]["url"],
            "score": float(normalized_score)
        })

    # Sort results by cross-encoder score in descending order
    results = sorted(results, key=lambda x: x["score"], reverse=True)[:10]
    return results


def main():
    """
    Main function to process URLs, prepare for RAG, and demonstrate a query.
    """
    urls = ["https://planet.python.org/"]
    index, embeddings, model = prepare_for_rag(urls, batch_size=32)

    if index is None or not embeddings:
        print("No data indexed, exiting.")
        return

    # Example query
    query_text = "What is Python programming?"
    results = query_rag(index, embeddings, model, query_text, k=20)

    print("\nQuery Results:")
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Header: {result['header'] or 'No header'}")
        print(f"Text: {result['text'][:200]}...")  # Truncate for display
        print(f"URL: {result['url']}")
        print(f"Score: {result['score']:.4f}")


if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     query_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_isekai_anime_2025/query.md"
#     html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_isekai_anime_2025/pages/animebytes_in_15_best_upcoming_isekai_anime_in_2025/page.html"
