from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from jet.data.header_types import TextNode
from jet.data.header_utils import VectorStore
from jet.models.embeddings.base import load_embed_model
from jet.models.model_types import EmbedModelType
from jet.logger import logger


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(similarity)


def search_headers(
    query: str,
    vector_store: 'VectorStore',
    model: EmbedModelType = "all-MiniLM-L6-v2",
    top_k: int = 5
) -> List[Tuple[TextNode, float]]:
    """Search for top-k relevant nodes based on query embedding."""
    logger.debug(f"Searching for query: {query}")
    transformer = load_embed_model(model)
    query_embedding = transformer.encode([query], show_progress_bar=False)[0]
    embeddings = vector_store.get_embeddings()
    nodes = vector_store.get_nodes()

    if not embeddings.size:
        logger.warning("Empty vector store, returning empty results")
        return []

    similarities = []
    for node in nodes:
        header_prefix = f"{node.header}\n" if node.header else ""
        content = node.content
        if header_prefix and content.startswith(header_prefix.strip()):
            content = content[len(header_prefix):].strip()

        # Check for empty content
        if not content.strip():
            similarities.append(0.0)
            continue

        # Calculate similarity for content
        content_embedding = transformer.encode(
            [content], show_progress_bar=False)[0]
        content_sim = cosine_similarity(query_embedding, content_embedding)

        # Calculate similarity for header (parent_header + header if parent exists, else just header)
        header_text = f"{node.parent_header}\n{node.header}" if node.parent_header and node.header else node.header or ""
        header_sim = 0.0
        if header_text:
            header_embedding = transformer.encode(
                [header_text], show_progress_bar=False)[0]
            header_sim = cosine_similarity(query_embedding, header_embedding)

        # Average the similarities (based on available components)
        sim_count = sum(1 for sim in [content_sim, header_sim] if sim > 0)
        final_sim = sum([content_sim, header_sim]) / max(sim_count, 1)
        similarities.append(final_sim)

    top_k_indices = np.argsort(similarities)[-top_k:][::-1]
    results = []
    for i in top_k_indices:
        if similarities[i] <= 0:
            continue
        node = nodes[i]
        header_prefix = f"{node.header}\n" if node.header else ""
        content = node.content
        if header_prefix and content.startswith(header_prefix.strip()):
            content = content[len(header_prefix):].strip()
        adjusted_node = TextNode(
            id=node.id,
            line=node.line,
            type=node.type,
            header=node.header,
            content=content,
            meta=node.meta,
            parent_id=node.parent_id,
            parent_header=None if not node.parent_header else node.parent_header,
            chunk_index=node.chunk_index,
            num_tokens=node.num_tokens
        )
        results.append((adjusted_node, similarities[i]))

    logger.debug(f"Found {len(results)} relevant nodes for query")
    return results
