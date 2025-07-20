import logging
import numpy as np

from typing import List, Literal, Optional
from numpy.typing import NDArray
from chromadb.api.types import QueryResult

from jet.db.chroma import SearchResult
from jet.data.utils import generate_key
from jet.logger import logger
from jet.transformers.object import make_serializable


def calculate_vector_scores(distances: List[float]) -> List[float]:
    """
    Transform cosine distances (1 - cosine_similarity) into Similarity scores in [0, 1].

    Args:
        distances: List of cosine distances from vector search (smaller is more similar)

    Returns:
        List of similarity scores in [0, 1], where higher scores indicate closer matches
    """
    if not distances:
        logger.debug("Empty distance list provided, returning empty scores")
        return []

    # Convert to numpy array for efficient computation
    distances_array = np.array(distances, dtype=np.float64)
    logger.debug("Input cosine distances: %s", distances_array.tolist())

    # Transform cosine distance (1 - cosine_similarity) to similarity
    scores = 1 - distances_array

    # Clip scores to [0, 1] to handle any numerical errors
    scores = np.clip(scores, 0, 1)
    scores_list = scores.tolist()

    logger.debug("Transformed similarity scores: %s", scores_list)
    return scores_list


def convert_search_results(query_result: QueryResult) -> list[SearchResult]:
    # Initialize an empty list to hold the converted query_result
    converted_results = []

    query_result = make_serializable(query_result)

    # Extract the values from the query_result dictionary
    documents = query_result.get('documents', [])[0]
    metadatas = query_result.get('metadatas', [])[0]
    distances = query_result.get('distances', [])[0]
    ids = query_result.get('ids', [])[0] if 'ids' in query_result else [
        doc.id for doc in documents]

    # Calculate the scores using the calculate_vector_scores function
    scores = calculate_vector_scores(distances)

    # Iterate over the values and build the new format
    for i, doc in enumerate(documents):
        converted_result = {
            'id': ids[i],
            'document': doc,
            # Use empty dict if metadata is None
            'metadata': metadatas[i],
            'score': scores[i]  # Use the calculated score
        }
        converted_results.append(converted_result)

    # Sort the results by score in reverse order (highest score first) using sorted
    return sorted(converted_results, key=lambda x: x['score'], reverse=True)
