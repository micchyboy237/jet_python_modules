import re
from typing import List, Optional, Set, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from jet.data.header_types import NodeWithScore, TextNode
from jet.data.header_utils import VectorStore
from jet.models.embeddings.base import generate_embeddings, load_embed_model
from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry
from jet.models.model_types import EmbedModelType
from jet.logger import logger


def preprocess_text(
    text: str,
    preserve_chars: Optional[Set[str]] = None,
    remove_stopwords: bool = False,
    apply_lemmatization: bool = False
) -> str:
    """
    Preprocess the input text with configurable options for normalization.

    Args:
        text (str): The input text to preprocess.
        preserve_chars (Optional[Set[str]]): Set of special characters to preserve (e.g., {'-', '_'}).
        remove_stopwords (bool): Whether to remove common stopwords (not implemented in this version).
        apply_lemmatization (bool): Whether to apply lemmatization (not implemented in this version).

    Returns:
        str: The preprocessed text.
    """
    if not text or not text.strip():
        logger.debug(f"Empty or whitespace-only input text: '{text}'")
        return ""

    # Log original text for debugging
    logger.debug(f"Preprocessing text: '{text}'")

    # Step 1: Normalize whitespace (replace multiple spaces, tabs, newlines with single space)
    text = re.sub(r'\s+', ' ', text.strip())

    # Step 2: Handle common contractions
    contractions = {
        "aren't": "are not",
        "can't": "cannot",
        "couldn't": "could not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'll": "he will",
        "he's": "he is",
        "i'd": "i would",
        "i'll": "i will",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it's": "it is",
        "let's": "let us",
        "mightn't": "might not",
        "mustn't": "must not",
        "shan't": "shall not",
        "she'd": "she would",
        "she'll": "she will",
        "she's": "she is",
        "shouldn't": "should not",
        "that's": "that is",
        "there's": "there is",
        "they'd": "they would",
        "they'll": "they will",
        "they're": "they are",
        "they've": "they have",
        "we'd": "we would",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "where's": "where is",
        "who'd": "who would",
        "who'll": "who will",
        "who're": "who are",
        "who's": "who is",
        "who've": "who have",
        "won't": "will not",
        "wouldn't": "would not",
        "you'd": "you would",
        "you'll": "you will",
        "you're": "you are",
        "you've": "you have",
        "'s": " is",  # Simple possessive handling
        "'re": " are",
        "'ve": " have",
        "'d": " would",
        "'ll": " will",
        "n't": " not"
    }
    for contraction, expanded in contractions.items():
        text = re.sub(r'\b' + contraction + r'\b',
                      expanded, text, flags=re.IGNORECASE)

    # Step 3: Convert to lowercase
    text = text.lower()

    # Step 4: Remove special characters, preserving specified ones
    # Default to preserving hyphens and underscores
    preserve_chars = preserve_chars or {'-', '_'}
    # Create regex pattern: keep alphanumeric, spaces, and preserved characters
    pattern = r'[^a-z0-9\s' + ''.join(map(re.escape, preserve_chars)) + r']'
    text = re.sub(pattern, '', text)

    # Step 5: Normalize whitespace again after character removal
    text = re.sub(r'\s+', ' ', text.strip())

    # Note: Stopword removal and lemmatization are not implemented in this version
    # To add lemmatization, use spacy: nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    # For stopwords, use nltk.corpus.stopwords or a custom list
    if remove_stopwords:
        logger.warning("Stopword removal not implemented in this version")
    if apply_lemmatization:
        logger.warning("Lemmatization not implemented in this version")

    logger.debug(f"Preprocessed text: '{text}'")
    return text


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(similarity)


def calculate_similarity_scores(query: str, nodes: List[TextNode], model: EmbedModelType, batch_size: int = 32) -> List[float]:
    # Preprocess the query for embedding
    preprocessed_query = preprocess_text(query)
    query_embedding = SentenceTransformerRegistry.generate_embeddings(
        [preprocessed_query], return_format="numpy")[0]

    header_texts = []
    original_header_texts = []
    content_texts = []
    original_content_texts = []
    header_prefixes = []

    for node in nodes:
        # Get original and preprocessed header
        header_text = f"{' '.join(node.get_parent_headers())}\n{node.header}" if node.parent_header else node.header
        preprocessed_header = preprocess_text(
            header_text) if header_text else ""
        original_header_texts.append(header_text)
        header_texts.append(preprocessed_header)

        # Get original and preprocessed content
        header_prefix = f"{node.header}\n" if node.header else ""
        content = node.content
        original_content = content
        if header_prefix and content.startswith(header_prefix.strip()):
            content = content[len(header_prefix):].strip()
        preprocessed_content = preprocess_text(content) if content else ""
        content_texts.append(preprocessed_content)
        original_content_texts.append(original_content)
        header_prefixes.append(header_prefix)

    # Combine preprocessed texts for embedding
    all_texts = [text for text in header_texts + content_texts if text.strip()]
    all_embeddings = SentenceTransformerRegistry.generate_embeddings(
        all_texts, batch_size=batch_size, show_progress=True, return_format="numpy")

    # Split embeddings back into headers and content
    header_embeddings = all_embeddings[:len(header_texts)]
    content_embeddings = all_embeddings[len(header_texts):]

    similarities = []
    header_idx = 0
    content_idx = 0

    for i, node in enumerate(nodes):
        header_text = header_texts[i]
        content = content_texts[i]

        # Initialize similarities
        header_sim = 0.0
        content_sim = 0.0
        sim_count = 0

        # Calculate header similarity if header exists
        if header_text.strip():
            header_embedding = header_embeddings[header_idx]
            header_sim = cosine_similarity(query_embedding, header_embedding)
            sim_count += 1
            header_idx += 1

        # Calculate content similarity if content exists
        if content.strip():
            content_embedding = content_embeddings[content_idx]
            content_sim = cosine_similarity(query_embedding, content_embedding)
            sim_count += 1
            content_idx += 1

        # Compute final similarity with penalty for single-component matches
        final_sim = sum([content_sim, header_sim]) / max(sim_count, 1)
        if sim_count == 1:
            final_sim *= 0.5  # Apply penalty to reduce score
        similarities.append(final_sim)

        # Update node metadata with original texts
        node.metadata = {
            "sim_count": sim_count,
            "header_similarity": header_sim,
            "content_similarity": content_sim,
            "header_text": original_header_texts[i],
            "content": original_content_texts[i],
        }

    return similarities


def search_headers(
    query: str,
    vector_store: 'VectorStore',
    model: EmbedModelType = "all-MiniLM-L6-v2",
    top_k: Optional[int] = 10,
    threshold: float = 0.0
) -> List[NodeWithScore]:
    """Search for top-k relevant nodes based on query embedding, filtering by similarity threshold."""
    logger.debug(f"Searching for query: {query} with threshold: {threshold}")
    embeddings = vector_store.get_embeddings()
    nodes = vector_store.get_nodes()
    if not top_k:
        top_k = len(nodes)
    if not embeddings.size:
        logger.warning("Empty vector store, returning empty results")
        return []
    similarities = calculate_similarity_scores(query, nodes, model)
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]
    results = []
    for rank, i in enumerate(top_k_indices, 1):
        if similarities[i] <= threshold:
            continue
        node = nodes[i]
        header_prefix = f"{node.header}\n" if node.header else ""
        content = node.content
        if header_prefix and content.startswith(header_prefix.strip()):
            content = content[len(header_prefix):].strip()
        adjusted_node = NodeWithScore(
            id=node.id,
            doc_index=node.doc_index,
            line=node.line,
            type=node.type,
            header=node.header,
            content=content,
            meta=node.meta,
            parent_id=node.parent_id,
            parent_header=None if not node.parent_header else node.parent_header,
            chunk_index=node.chunk_index,
            num_tokens=node.num_tokens,
            doc_id=node.doc_id,
            metadata=node.metadata,
            rank=rank,
            score=similarities[i],
        )
        results.append(adjusted_node)
    logger.debug(
        f"Found {len(results)} relevant nodes for query after threshold {threshold}")
    return results
