from typing import List, Dict, Optional
from jet.data.header_types import NodeType, Nodes, TextNode
from jet.models.embeddings.base import load_embed_model
from jet.models.model_types import EmbedModelType
from jet.models.tokenizer.base import get_tokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
from jet.logger import logger


class VectorStore:
    """In-memory vector store for RAG embeddings."""

    def __init__(self):
        self.embeddings: List[np.ndarray] = []
        self.nodes: List[TextNode] = []

    def add(self, node: TextNode, embedding: np.ndarray) -> None:
        """Add a node and its embedding to the store."""
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
    batch_size: int = 32
) -> VectorStore:
    """Prepare nodes for RAG by generating embeddings for their content."""
    logger.debug(f"Preparing {len(nodes)} nodes for RAG with model {model}")
    transformer = load_embed_model(model)
    vector_store = VectorStore()

    # Prepare texts for embedding (parent_header + header + content)
    texts = []
    for node in nodes:
        text_parts = []
        if node.parent_header:
            text_parts.append(node.parent_header)
        text_parts.append(node.header)
        text_parts.append(node.content)
        texts.append("\n".join(text_parts))

    # Generate embeddings in batches
    embeddings = transformer.encode(
        texts, batch_size=batch_size, show_progress_bar=False)

    # Store embeddings and nodes
    for node, embedding in zip(nodes, embeddings):
        vector_store.add(node, embedding)

    return vector_store
