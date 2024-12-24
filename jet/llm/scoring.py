import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def calculate_vector_scores(distances: list[float]) -> list[float]:
    return [1 - distance for distance in distances]


def calculate_vector_scores_cosine(query_vector: list[float | int] = [], document_embeddings: list[list[dict]] = [], distances: list[list[float | int]] = [], documents: list[list[str]] = []) -> list[float | dict]:
    # Convert query vector into a 2D array for cosine similarity calculation
    query_vector = np.array(query_vector).reshape(1, -1)

    scores = []
    # If there are distances, you can use those directly, but here we're calculating based on vectors
    if distances:
        # Assuming the first list of distances
        distances = np.array(distances[0])
        # Convert to similarity scores if needed (higher distance = lower similarity)
        # Simple transformation (inversely related)
        score = 1 / (1 + distances)
        texts_with_scores = [{"text": documents[0][idx], "score": value}
                             for idx, value in enumerate(score)]
        scores.append(texts_with_scores)
    else:
        texts = []
        embeddings_matrix = []
        for document_embedding in document_embeddings:
            texts.append(document_embedding['text'])
            embeddings_matrix.append(document_embedding['embeddings'])
        # Use cosine similarity if no pre-calculated distances are available
        document_vectors = np.array(embeddings_matrix)
        similarity = cosine_similarity(query_vector, document_vectors)
        score = similarity[0]
        texts_with_scores = [{"text": texts[idx], "score": value}
                             for idx, value in enumerate(score)]
        scores.append(texts_with_scores)

    return scores
