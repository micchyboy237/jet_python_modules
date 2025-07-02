from typing import List, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from jet.data.header_types import NodeWithScore, TextNode
from jet.data.header_utils import VectorStore
from jet.models.embeddings.base import generate_embeddings, load_embed_model
from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry
from jet.models.model_types import EmbedModelType
from jet.logger import logger


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(similarity)


def calculate_similarity_scores(query: str, nodes: List[TextNode], model: EmbedModelType, batch_size: int = 32) -> List[float]:
    # transformer = load_embed_model(model)
    # query_embedding = transformer.encode([query], show_progress_bar=False)[0]
    registry = SentenceTransformerRegistry()
    registry.load_model(model)
    query_embedding = registry.generate_embeddings(
        [query], return_format="numpy")[0]

    header_texts = [
        f"{"\n".join(node.get_parent_headers())}\n{node.header}" if node.parent_header else node.header for node in nodes]
    content_texts = []
    header_prefixes = []
    for node in nodes:
        header_prefix = f"{node.header}\n" if node.header else ""
        content = node.content
        if header_prefix and content.startswith(header_prefix.strip()):
            content = content[len(header_prefix):].strip()
        content_texts.append(content)
        header_prefixes.append(header_prefix)

    all_texts = [text for text in header_texts + content_texts if text.strip()]
    all_embeddings = registry.generate_embeddings(
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
        # Only one component contributed (other is 0.0 or empty)
        if sim_count == 1:
            final_sim *= 0.5  # Apply penalty to reduce score
        similarities.append(final_sim)

        # Update node metadata
        node.metadata = {
            "sim_count": sim_count,
            "header_similarity": header_sim,
            "content_similarity": content_sim,
            "header_text": header_text,
            "content": content,
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
