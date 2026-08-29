# rag_module_v1/retrieval.py

from collections import defaultdict

from jet.adapters.llama_cpp.embed_utils import embed
from jet.adapters.llama_cpp.rerank_utils import rerank
from jet.adapters.llama_cpp.scoring_utils import cosine_similarity
from jet.vectors.reranker.bm25 import rerank_bm25

from .schemas import Chunk, RetrievedChunk


def vector_retrieve(
    query: str,
    chunks: list[Chunk],
    top_k: int,
    min_score: float | None = None,
) -> list[RetrievedChunk]:
    if not chunks:
        return []

    texts = [c.content for c in chunks]
    query_emb = embed(query)
    doc_embs = embed(texts, show_progress=False)

    scored = []
    for chunk, doc_emb in zip(chunks, doc_embs):
        score = float(cosine_similarity(query_emb, doc_emb))
        if min_score is None or score >= min_score:
            scored.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=score,
                    vector_score=score,
                    arms=["vector"],
                )
            )

    scored.sort(key=lambda r: r.score, reverse=True)
    return scored[:top_k]


def bm25_retrieve(
    query: str,
    chunks: list[Chunk],
    top_k: int,
    min_score: float = 0.0,
) -> list[RetrievedChunk]:
    if not chunks:
        return []

    documents = [c.content for c in chunks]
    ids = [c.chunk_id for c in chunks]
    metadatas = [c.metadata for c in chunks]

    _, results = rerank_bm25(
        query=query,
        documents=documents,
        ids=ids,
        metadatas=metadatas,
    )

    out: list[RetrievedChunk] = []

    chunk_by_id = {c.chunk_id: c for c in chunks}

    for r in results:
        score = float(r["score"])
        if score < min_score:
            continue

        chunk = chunk_by_id[r["id"]]
        out.append(
            RetrievedChunk(
                chunk=chunk,
                score=score,
                bm25_score=score,
                arms=["bm25"],
            )
        )

    out.sort(key=lambda r: r.score, reverse=True)
    return out[:top_k]


def rrf_fusion(
    vector_results: list[RetrievedChunk],
    bm25_results: list[RetrievedChunk],
    *,
    k: int = 60,
    top_k: int = 20,
) -> list[RetrievedChunk]:
    fused: dict[str, RetrievedChunk] = {}
    rrf_scores: dict[str, float] = defaultdict(float)

    for rank, result in enumerate(vector_results, start=1):
        cid = result.chunk.chunk_id
        rrf_scores[cid] += 1.0 / (k + rank)

        if cid not in fused:
            fused[cid] = result
        else:
            fused[cid].vector_score = result.vector_score
            fused[cid].arms = sorted(set(fused[cid].arms + ["vector"]))

    for rank, result in enumerate(bm25_results, start=1):
        cid = result.chunk.chunk_id
        rrf_scores[cid] += 1.0 / (k + rank)

        if cid not in fused:
            fused[cid] = result
        else:
            fused[cid].bm25_score = result.bm25_score
            fused[cid].arms = sorted(set(fused[cid].arms + ["bm25"]))

    for cid, result in fused.items():
        result.score = rrf_scores[cid]

    results = list(fused.values())
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]


def rerank_chunks(
    query: str,
    candidates: list[RetrievedChunk],
    top_n: int,
) -> list[RetrievedChunk]:
    if not candidates:
        return []

    documents = [r.chunk.content for r in candidates]

    reranked = rerank(
        query=query,
        documents=documents,
        top_n=min(top_n, len(documents)),
        method="auto",
        normalize_scores=True,
    )

    out: list[RetrievedChunk] = []

    for rr in reranked:
        original = candidates[rr["index"]]
        original.rerank_score = float(rr["score"])
        original.score = float(rr["score"])
        out.append(original)

    out.sort(key=lambda r: r.score, reverse=True)
    return out
