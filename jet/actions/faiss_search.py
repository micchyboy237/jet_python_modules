from typing import Optional
from jet.llm.helpers.faiss_utils import (
    get_faiss_model,
    create_faiss_index,
    search_faiss_index,
)

# Define the typed dict for each result
Result = dict[str, str | float]


def faiss_search(
    queries: list[str],
    candidates: list[str],
    *,
    top_k: Optional[int] = 3,
    nlist: Optional[int] = None,  # Make nlist optional and dynamic
):
    model = get_faiss_model()

    candidate_embeddings = model.encode(candidates)
    query_embeddings = model.encode(queries)

    d = candidate_embeddings.shape[1]

    # Dynamically adjust nlist: FAISS recommends at least 39x data points
    num_candidates = len(candidates)
    # if nlist is None:
    #     nlist = max(1, min(100, num_candidates // 39))  # Ensure nlist is valid
    if nlist is None:
        # Use sqrt rule: FAISS suggests sqrt(num_candidates) for optimal nlist
        nlist = max(1, min(int(num_candidates ** 0.5), num_candidates // 39))

    index = create_faiss_index(candidate_embeddings, d, nlist)

    distances, indices = search_faiss_index(
        index, query_embeddings, top_k, nprobe=min(10, nlist)
    )  # Ensure nprobe does not exceed nlist

    # Update results to group by query
    results: dict[str, list[Result]] = {
        queries[i]: [
            {"text": candidates[indices[i][j]], "score": distances[i][j]}
            # Avoid index errors
            for j in range(top_k) if indices[i][j] < len(candidates)
        ]
        for i in range(len(queries))
    }

    return results
