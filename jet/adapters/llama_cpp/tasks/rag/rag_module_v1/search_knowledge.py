# rag_module_v1/search_knowledge.py

import json
import time
from pathlib import Path

from jet.logger import logger

from .config import RAGConfig
from .corpus import filter_chunks, load_corpus
from .formatting import format_context
from .query_processing import (
    extract_metadata,
    normalize_input,
    rewrite_query,
    validate_query,
)
from .retrieval import bm25_retrieve, rerank_chunks, rrf_fusion, vector_retrieve
from .schemas import SearchResult, SearchStatus

DEFAULT_CORPUS_PATH = Path(__file__).parent / "data" / "corpus_v1.jsonl"


class KnowledgeSearchTool:
    def __init__(
        self,
        corpus_path: str | Path = DEFAULT_CORPUS_PATH,
        config: RAGConfig | None = None,
    ):
        self.config = config or RAGConfig()
        self.chunks = load_corpus(corpus_path)

    def search_knowledge(self, query: str, thought_context: str = "") -> dict:
        start = time.perf_counter()
        trace = {
            "original_query": query,
            "thought_context_len": len(thought_context or ""),
        }

        try:
            query = validate_query(query, self.config.max_query_chars)
            thought_context = normalize_input(thought_context)[
                : self.config.max_thought_context_chars
            ]

            if self.config.enable_query_rewrite:
                query_used = rewrite_query(query, thought_context)
            else:
                query_used = query

            trace["query_used"] = query_used

            if self.config.enable_metadata_extraction:
                metadata_filters = extract_metadata(query_used)
            else:
                metadata_filters = {}

            trace["metadata_filters"] = metadata_filters

            filtered_chunks = filter_chunks(self.chunks, metadata_filters)
            trace["filtered_chunk_count"] = len(filtered_chunks)

            vector_results = vector_retrieve(
                query=query_used,
                chunks=filtered_chunks,
                top_k=self.config.vector_top_k,
                min_score=self.config.vector_min_score,
            )

            bm25_results = bm25_retrieve(
                query=query_used,
                chunks=filtered_chunks,
                top_k=self.config.bm25_top_k,
                min_score=self.config.bm25_min_score,
            )

            trace["vector_count"] = len(vector_results)
            trace["bm25_count"] = len(bm25_results)

            fused = rrf_fusion(
                vector_results,
                bm25_results,
                top_k=self.config.fusion_top_k,
            )

            trace["fused_count"] = len(fused)

            if self.config.enable_rerank:
                final_results = rerank_chunks(
                    query=query_used,
                    candidates=fused,
                    top_n=self.config.rerank_top_n,
                )
            else:
                final_results = fused[: self.config.rerank_top_n]

            threshold = self._compute_dynamic_threshold(final_results)
            trace["threshold"] = threshold

            valid_results = [r for r in final_results if r.score >= threshold]
            trace["valid_count"] = len(valid_results)

            latency_ms = int((time.perf_counter() - start) * 1000)

            if not valid_results:
                result = SearchResult(
                    status=SearchStatus.ABSTAINED,
                    answer_context="No relevant context found in the knowledge base.",
                    query_used=query_used,
                    metadata_applied=metadata_filters,
                    _latency_ms=latency_ms,
                )
            else:
                context, truncated = format_context(
                    valid_results,
                    max_tokens=self.config.max_context_tokens,
                )

                sources = [
                    {
                        "chunk_id": r.chunk.chunk_id,
                        "doc_id": r.chunk.doc_id,
                        "doc_title": r.chunk.doc_title,
                        "relevance_score": round(r.score, 4),
                        "vector_score": (
                            round(r.vector_score, 4)
                            if r.vector_score is not None
                            else None
                        ),
                        "bm25_score": (
                            round(r.bm25_score, 4) if r.bm25_score is not None else None
                        ),
                        "rerank_score": (
                            round(r.rerank_score, 4)
                            if r.rerank_score is not None
                            else None
                        ),
                        "arms": r.arms,
                        "content_preview": r.chunk.content[:160],
                    }
                    for r in valid_results
                ]

                result = SearchResult(
                    status=SearchStatus.FOUND,
                    answer_context=context,
                    sources=sources,
                    query_used=query_used,
                    metadata_applied=metadata_filters,
                    truncated=truncated,
                    _latency_ms=latency_ms,
                )

        except Exception as e:
            latency_ms = int((time.perf_counter() - start) * 1000)
            logger.exception(f"search_knowledge failed: {e}")

            result = SearchResult(
                status=SearchStatus.ERROR,
                answer_context=f"Knowledge search failed: {type(e).__name__}",
                query_used=query,
                _latency_ms=latency_ms,
            )
            trace["error"] = str(e)

        trace["status"] = result.status.value
        trace["latency_ms"] = result._latency_ms

        logger.info(json.dumps(trace, ensure_ascii=False))

        return result.to_dict(include_internal=True)

    def _compute_dynamic_threshold(self, results) -> float:
        if len(results) < 3:
            return self.config.default_abstention_threshold

        scores = [r.score for r in results]
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        std_score = variance**0.5

        if std_score < 1e-6:
            return max(
                mean_score - self.config.zero_variance_margin,
                self.config.min_absolute_threshold,
            )

        return max(
            mean_score + 0.5 * std_score,
            self.config.min_absolute_threshold,
        )


_tool: KnowledgeSearchTool | None = None


def get_tool() -> KnowledgeSearchTool:
    global _tool
    if _tool is None:
        _tool = KnowledgeSearchTool()
    return _tool


def search_knowledge(query: str, thought_context: str = "") -> dict:
    return get_tool().search_knowledge(query, thought_context)
