import re
from typing import List, Optional, Set
import numpy as np
from sentence_transformers import SentenceTransformer
from jet.data.header_types import NodeWithScore, TextNode
from jet.data.header_utils import VectorStore, preprocess_text
from jet.models.embeddings.base import generate_embeddings
from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry
from jet.models.model_types import EmbedModelType
from jet.logger import logger
from jet.utils.text_constants import TEXT_CONTRACTIONS_EN


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(similarity)


def calculate_similarity_scores(query: str, vector_store: VectorStore) -> List[float]:
    preprocessed_query = preprocess_text(query)
    query_embedding = SentenceTransformerRegistry.generate_embeddings(
        [preprocessed_query], return_format="numpy")[0]
    embeddings = vector_store.get_embeddings()
    processed_texts = vector_store.get_processed_texts()
    nodes = vector_store.get_nodes()
    similarities = []
    for i, (embedding, processed_text, node) in enumerate(zip(embeddings, processed_texts, nodes)):
        sim_count = 0
        similarity = 0.0
        if processed_text.strip():
            similarity = cosine_similarity(query_embedding, embedding)
            sim_count += 1
        final_sim = similarity / max(sim_count, 1)
        similarities.append(final_sim)
        node.metadata = {
            "sim_count": sim_count,
            "similarity": similarity,
            "processed_text": processed_text,
            "header_text": f"{' '.join(node.get_parent_headers())}\n{node.header}" if node.parent_header else node.header,
            "content": node.content,
        }
    return similarities


def search_headers(
    query: str,
    vector_store: VectorStore,
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
    similarities = calculate_similarity_scores(query, vector_store)
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
