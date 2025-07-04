import re
from typing import List, Dict, Optional, Set
from jet.data.header_types import NodeType, Nodes, TextNode
from jet.models.embeddings.base import generate_embeddings, load_embed_model
from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry
from jet.models.model_types import EmbedModelType
from jet.models.tokenizer.base import get_tokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
from jet.logger import logger
from jet.data.header_utils import split_and_merge_headers
from tokenizers import Tokenizer

from jet.utils.text_constants import TEXT_CONTRACTIONS_EN


class VectorStore:
    """In-memory vector store for RAG embeddings."""

    def __init__(self):
        self.embeddings: List[np.ndarray] = []
        self.nodes: List[TextNode] = []
        self.processed_texts: List[str] = []

    def add(self, node: TextNode, embedding: np.ndarray, processed_text: str) -> None:
        """Add a node, its embedding, and preprocessed text to the store."""
        logger.debug(
            f"Adding node {node.id} with content length {len(node.content)}, num_tokens {node.num_tokens}, doc_id={node.doc_id}")
        self.embeddings.append(embedding)
        self.nodes.append(node)
        self.processed_texts.append(processed_text)

    def get_nodes(self) -> List[TextNode]:
        """Return all stored nodes with original text."""
        return self.nodes

    def get_processed_texts(self) -> List[str]:
        """Return all preprocessed texts used for embeddings."""
        return self.processed_texts

    def get_embeddings(self) -> np.ndarray:
        """Return all embeddings as a NumPy array."""
        return np.array(self.embeddings)

    def get_embeddings_shape(self) -> tuple[int, int]:
        """Return the dimension of the embeddings."""
        return self.embeddings[0].shape


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
    for contraction, expanded in TEXT_CONTRACTIONS_EN.items():
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


def prepare_for_rag(
    nodes: Nodes,
    model: EmbedModelType = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    chunk_size: Optional[int] = None,
    chunk_overlap: int = 0,
    buffer: int = 0,
    tokenizer: Optional[Tokenizer] = None,
) -> VectorStore:
    """Prepare nodes for RAG by generating normalized embeddings for their preprocessed content, with optional chunking."""
    logger.debug(
        f"Preparing {len(nodes)} nodes for RAG with model {model}, chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, buffer={buffer}")
    if not tokenizer:
        tokenizer = SentenceTransformerRegistry.get_tokenizer()
    if chunk_size is not None:
        logger.debug(f"Applying chunking with chunk_size={chunk_size}")
        nodes = split_and_merge_headers(
            nodes,
            model=model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tokenizer=tokenizer,
            buffer=buffer
        )
        logger.debug(f"After chunking, received {len(nodes)} nodes")
        for node in nodes:
            logger.debug(
                f"Chunked node {node.id}: header={node.header}, content_length={len(node.content)}, num_tokens={node.num_tokens}, doc_id={node.doc_id}")
    vector_store = VectorStore()
    texts = []
    processed_texts = []
    for node in nodes:
        text_parts = []
        if node.parent_header and node.parent_header != node.header:
            text_parts.append(node.parent_header)
        text_parts.append(node.header)
        text_parts.append(node.content)
        text = "\n".join(part for part in text_parts if part)
        processed_text = preprocess_text(text)
        if node.num_tokens == 0:
            token_ids = tokenizer.encode(text, add_special_tokens=False).ids
            token_ids = [tid for tid in token_ids if tid != 0]
            node.num_tokens = len(token_ids)
            logger.debug(
                f"Calculated num_tokens for node {node.id}: text_length={len(text)}, num_tokens={node.num_tokens}")
        else:
            logger.debug(
                f"Using existing num_tokens for node {node.id}: num_tokens={node.num_tokens}")
        texts.append(text)
        processed_texts.append(processed_text)
    logger.debug(f"Encoding {len(texts)} texts for VectorStore")
    embeddings = SentenceTransformerRegistry.generate_embeddings(
        processed_texts, batch_size=batch_size, show_progress=True, return_format="numpy")
    # Normalize embeddings to unit vectors
    for i, embedding in enumerate(embeddings):
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embeddings[i] = embedding / norm
        else:
            logger.warning(
                f"Zero norm embedding for node {nodes[i].id}, keeping as is")
    for node, embedding, text, processed_text in zip(nodes, embeddings, texts, processed_texts):
        store_node = TextNode(
            id=node.id,
            doc_index=node.doc_index,
            line=node.line,
            type=node.type,
            header=node.header,
            content=node.content,
            meta=node.meta,
            parent_id=node.parent_id,
            parent_header=node.parent_header,
            chunk_index=node.chunk_index,
            num_tokens=node.num_tokens,
            doc_id=node.doc_id,
        )
        if node.parent_id:
            store_node._parent_node = node.get_parent_node()
        vector_store.add(store_node, embedding, processed_text)
    logger.debug(f"Vector store contains {len(vector_store.nodes)} nodes")
    return vector_store
