from typing import List, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from jet.data.header_types import NodeWithScore, TextNode
from jet.data.header_utils import VectorStore
from jet.models.embeddings.base import load_embed_model
from jet.models.model_types import EmbedModelType
from jet.logger import logger


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(similarity)


def calculate_similarity_scores(query: str, nodes: List[TextNode], model: EmbedModelType) -> List[float]:
    transformer = load_embed_model(model)
    query_embedding = transformer.encode([query], show_progress_bar=False)[0]
    similarities = []
    for node in nodes:
        header_prefix = f"{node.header}\n" if node.header else ""
        content = node.content
        if header_prefix and content.startswith(header_prefix.strip()):
            content = content[len(header_prefix):].strip()
        if not content.strip():
            similarities.append(0.0)
            continue
        content_embedding = transformer.encode(
            [content], show_progress_bar=False)[0]
        content_sim = cosine_similarity(query_embedding, content_embedding)
        header_text = f"{node.parent_header}\n{node.header}" if node.parent_header else node.header
        header_sim = 0.0
        header_embedding = transformer.encode(
            [header_text], show_progress_bar=False)[0]
        header_sim = cosine_similarity(query_embedding, header_embedding)
        sim_count = sum(1 for sim in [content_sim, header_sim] if sim > 0)
        final_sim = sum([content_sim, header_sim]) / max(sim_count, 1)
        similarities.append(final_sim)
        node.metadata = {
            "sim_count": sim_count,
            "header_similarity": header_sim,
            "content_similarity": content_sim,
        }
    return similarities


def search_headers(
    query: str,
    vector_store: 'VectorStore',
    model: EmbedModelType = "all-MiniLM-L6-v2",
    top_k: Optional[int] = 10
) -> List[NodeWithScore]:
    """Search for top-k relevant nodes based on query embedding."""
    logger.debug(f"Searching for query: {query}")
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
        if similarities[i] <= 0:
            continue
        node = nodes[i]
        header_prefix = f"{node.header}\n" if node.header else ""
        content = node.content
        if header_prefix and content.startswith(header_prefix.strip()):
            content = content[len(header_prefix):].strip()
        adjusted_node = NodeWithScore(
            id=node.id,
            line=node.line,
            type=node.type,
            header=node.header,
            content=content,
            meta=node.meta,
            parent_id=node.parent_id,
            parent_header=None if not node.parent_header else node.parent_header,
            chunk_index=node.chunk_index,
            num_tokens=node.num_tokens,
            doc_id=node.doc_id,  # Propagate required doc_id
            metadata=node.metadata,
            rank=rank,
            score=similarities[i],
        )
        results.append(adjusted_node)
    logger.debug(f"Found {len(results)} relevant nodes for query")
    return results
