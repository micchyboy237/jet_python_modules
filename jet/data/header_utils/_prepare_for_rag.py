import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from typing import Optional, Set
import logging
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
from jet.scrapers.utils import clean_newlines, clean_punctuations, clean_spaces
from jet.search.formatters import clean_string
from jet.utils.text_constants import TEXT_CONTRACTIONS_EN
from jet.data.header_utils._base import create_text_node, chunk_content, merge_nodes
from jet.wordnet.words import get_words

# Download required NLTK data (only needs to be done once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')


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
    remove_stopwords: bool = True,
    apply_lemmatization: bool = False
) -> str:
    """
    Preprocesses input text by normalizing whitespace, expanding contractions,
    converting to lowercase, removing unwanted characters, and optionally
    removing stopwords and applying lemmatization.

    Args:
        text: Input text to preprocess
        preserve_chars: Set of characters to preserve in the output
        remove_stopwords: Whether to remove common stopwords
        apply_lemmatization: Whether to apply lemmatization to words

    Returns:
        Preprocessed text as a string
    """
    if not text or not text.strip():
        return ""

    # Expand contractions
    for contraction, expanded in TEXT_CONTRACTIONS_EN.items():
        text = re.sub(r'\b' + contraction + r'\b',
                      expanded, text, flags=re.IGNORECASE)

    # Convert to lowercase
    text = text.lower()

    # Clean text
    text = clean_newlines(text, max_newlines=1)
    text = clean_punctuations(text)
    text = clean_spaces(text)
    text = clean_string(text)

    # Split into lines (if any)
    lines = text.splitlines() if '\n' in text else [text]
    preprocessed_lines = []
    stop_words = set(stopwords.words('english')) if remove_stopwords else set()

    for line in lines:
        words = get_words(line)
        if remove_stopwords:
            words = [word for word in words if word.lower() not in stop_words]
        preprocessed_lines.append(' '.join(words))

    text = '\n'.join(preprocessed_lines)

    # Apply lemmatization if specified
    if apply_lemmatization:
        lemmatizer = WordNetLemmatizer()
        words = []
        for line in preprocessed_lines:
            words.extend(line.split())
        words = [lemmatizer.lemmatize(word) for word in words]
        # Join words back into a string and normalize whitespace
        text = ' '.join(words)

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

    # if chunk_size is not None:
    #     logger.debug(f"Applying chunking with chunk_size={chunk_size}")
    #     nodes = merge_nodes(
    #         nodes=nodes,
    #         tokenizer=tokenizer,
    #         max_tokens=chunk_size,
    #         buffer=buffer
    #     )
    #     logger.debug(f"After merging, received {len(nodes)} nodes")
    #     for node in nodes:
    #         logger.debug(
    #             f"Merged node {node.id}: header={node.header}, content_length={len(node.content)}, num_tokens={node.num_tokens}, doc_id={node.doc_id}")

    vector_store = VectorStore()
    texts = []
    processed_texts = []
    for node in nodes:
        text_parts = []
        # text_parts.extend(node.get_parent_headers())
        if node.parent_header and node.parent_header != node.header:
            text_parts.append(node.parent_header)
        text_parts.append(node.header)
        text_parts.append(node.content)
        text = "\n".join(part for part in text_parts if part)
        processed_text = preprocess_text(text)
        if node.num_tokens == 0:
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            token_ids = token_ids if isinstance(
                token_ids, list) else token_ids.ids
            node.num_tokens = len([tid for tid in token_ids if tid != 0])
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
            meta=getattr(node, 'meta', {}),
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
