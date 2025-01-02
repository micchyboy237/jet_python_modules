import faiss
from jet.llm.helpers.faiss_utils import (
    get_model,
    create_faiss_index,
    search_faiss_index,
)
from jet.logger import logger, time_it

# Define the typed dict for each result
Result = dict[str, str | float]


def faiss_search(
    queries: list[str],
    candidates: list[str],
    *,
    top_k=3,
    nlist=100,
):
    model = time_it(get_model)()

    candidate_embeddings = model.encode(candidates)
    query_embeddings = model.encode(queries)

    d = candidate_embeddings.shape[1]

    top_k = 3  # Number of nearest neighbors
    index = create_faiss_index(candidate_embeddings, d, nlist)

    distances, indices = search_faiss_index(
        index, query_embeddings, top_k, nprobe=10)

    # Update results to group by query
    results: dict[str, list[Result]] = {
        queries[i]: [
            {"text": candidates[indices[i][j]], "score": distances[i][j]}
            for j in range(top_k)
        ]
        for i, query in enumerate(queries)
    }

    return results
