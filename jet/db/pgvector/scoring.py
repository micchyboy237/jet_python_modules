from chromadb.api.types import QueryResult
from jet.db.chroma import SearchResult
from jet.data.utils import generate_key
from jet.transformers import make_serializable


def calculate_vector_scores(distances: list[float]) -> list[float]:
    return [1 - distance for distance in distances]


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
