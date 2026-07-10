import requests

RERANK_URL = "http://192.168.68.150:8082/v1/rerank"
MODEL = "bge-reranker-v2-m3-Q4_K_M.gguf"


def rerank(query: str, documents: list[str], top_n: int | None = None) -> list[dict]:
    """
    Call the local llama.cpp reranker and return documents sorted by relevance.
    Each returned dict: {"document": str, "score": float, "index": int}
    """
    payload = {"model": MODEL, "query": query, "documents": documents}
    if top_n is not None:
        payload["top_n"] = top_n

    resp = requests.post(RERANK_URL, json=payload, timeout=30)
    resp.raise_for_status()
    results = resp.json()["results"]

    # attach back the original text and sort by score, highest first
    ranked = [
        {
            "document": documents[r["index"]],
            "score": r["relevance_score"],
            "index": r["index"],
        }
        for r in results
    ]
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


if __name__ == "__main__":
    query = "What is panda?"
    docs = [
        "hi",
        "it is a bear",
        "The giant panda is a bear species endemic to China.",
    ]
    for r in rerank(query, docs, top_n=3):
        print(f"{r['score']:.3f}  {r['document']}")
