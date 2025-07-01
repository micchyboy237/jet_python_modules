from typing import List, Dict, Optional
from jet.data.header_types import NodeType, Nodes, TextNode
from jet.models.embeddings.base import load_embed_model
from jet.models.model_types import EmbedModelType
from jet.models.tokenizer.base import get_tokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
from jet.logger import logger
from jet.data.header_utils import split_and_merge_headers
from tokenizers import Tokenizer


class VectorStore:
    """In-memory vector store for RAG embeddings."""

    def __init__(self):
        self.embeddings: List[np.ndarray] = []
        self.nodes: List[TextNode] = []

    def add(self, node: TextNode, embedding: np.ndarray) -> None:
        """Add a node and its embedding to the store."""
        logger.debug(
            f"Adding node {node.id} with content length {len(node.content)} and num_tokens {node.num_tokens}")
        self.embeddings.append(embedding)
        self.nodes.append(node)

    def get_nodes(self) -> List[TextNode]:
        """Return all stored nodes."""
        return self.nodes

    def get_embeddings(self) -> np.ndarray:
        """Return all embeddings as a NumPy array."""
        return np.array(self.embeddings)


def prepare_for_rag(
    nodes: Nodes,
    model: EmbedModelType = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    chunk_size: Optional[int] = None,
    chunk_overlap: int = 0,
    buffer: int = 0,
    tokenizer: Optional[Tokenizer] = None
) -> VectorStore:
    """Prepare nodes for RAG by generating embeddings for their content, with optional chunking."""
    logger.debug(
        f"Preparing {len(nodes)} nodes for RAG with model {model}, chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, buffer={buffer}")
    if not tokenizer:
        tokenizer = get_tokenizer(model)
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
                f"Chunked node {node.id}: header={node.header}, content_length={len(node.content)}, num_tokens={node.num_tokens}")
    vector_store = VectorStore()
    transformer = load_embed_model(model)
    texts = []
    for node in nodes:
        text_parts = []
        if node.parent_header and node.parent_header != node.header:
            text_parts.append(node.parent_header)
        text_parts.append(node.header)
        text_parts.append(node.content)
        text = "\n".join(part for part in text_parts if part)
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
    logger.debug(f"Encoding {len(texts)} RectorStore")
    embeddings = transformer.encode(
        texts, batch_size=batch_size, show_progress_bar=False)
    for node, embedding in zip(nodes, embeddings):
        vector_store.add(node, embedding)
    logger.debug(f"Vector store contains {len(vector_store.nodes)} nodes")
    return vector_store
