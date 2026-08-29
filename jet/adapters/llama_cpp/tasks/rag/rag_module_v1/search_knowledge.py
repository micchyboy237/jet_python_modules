import json
import logging
import time
from dataclasses import asdict, dataclass
from enum import Enum

# Configure structured logging for traceability
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("rag_tool")


class SearchStatus(str, Enum):
    FOUND = "found"
    ABSTAINED = "abstained"
    ERROR = "error"


@dataclass
class SearchResult:
    """Strict output contract for ReAct agent consumption."""

    status: SearchStatus
    answer_context: str = ""
    sources: list[dict] = None
    query_used: str = ""
    metadata_applied: dict = None
    _latency_ms: int = 0  # Internal metric for eval harness

    def __post_init__(self):
        if self.sources is None:
            self.sources = []
        if self.metadata_applied is None:
            self.metadata_applied = {}

    def to_dict(self) -> dict:
        """Agent-safe serialization. Excludes internal metrics."""
        d = asdict(self)
        d.pop("_latency_ms", None)
        return d


class KnowledgeSearchTool:
    def __init__(self, vector_store, bm25_index, reranker, rewriter_llm, config: dict):
        self.vs = vector_store
        self.bm25 = bm25_index
        self.reranker = reranker
        self.rewriter = rewriter_llm
        self.config = config  # Contains thresholds, top_k values, etc.

    def search_knowledge(self, query: str, thought_context: str = "") -> dict:
        """
        Primary entry point. Returns dict matching SearchResult schema.
        Instruments latency and logs full trace for eval/debugging.
        """
        start_time = time.perf_counter()
        trace = {"original_query": query, "thought_context": thought_context}

        try:
            # === PHASE 1: PRE-RETRIEVAL ===
            rewritten_query = self._rewrite_query(query, thought_context)
            trace["rewritten_query"] = rewritten_query

            metadata_filters = self._extract_metadata(rewritten_query)
            trace["metadata_filters"] = metadata_filters

            # === PHASE 2: HYBRID RETRIEVAL ===
            # CRITICAL: Apply filters BEFORE retrieval, threshold PER ARM before fusion
            vector_results = self.vs.search(
                query=rewritten_query,
                top_k=self.config["vector_top_k"],
                filters=metadata_filters,
                min_score=self.config["vector_min_score"],  # Per-arm threshold
            )

            bm25_results = self.bm25.search(
                query=rewritten_query,
                top_k=self.config["bm25_top_k"],
                filters=metadata_filters,
                min_score=self.config["bm25_min_score"],  # Per-arm threshold
            )

            trace["vector_count"] = len(vector_results)
            trace["bm25_count"] = len(bm25_results)

            # Detect silent arm failures
            if len(vector_results) == 0 and len(bm25_results) > 0:
                logger.warning(
                    f"Vector arm returned 0 results for query: {rewritten_query}"
                )
            if len(bm25_results) == 0 and len(vector_results) > 0:
                logger.warning(
                    f"BM25 arm returned 0 results for query: {rewritten_query}"
                )

            fused_results = self._rrf_fusion(vector_results, bm25_results)
            trace["fused_count"] = len(fused_results)

            # === PHASE 3: POST-RETRIEVAL VALIDATION ===
            reranked_results = self.reranker.rerank(
                query=rewritten_query,
                documents=fused_results,
                top_n=self.config["rerank_top_n"],
            )

            # Dynamic relevance floor
            threshold = self._compute_dynamic_threshold(reranked_results)
            trace["rerank_threshold"] = threshold

            valid_results = [r for r in reranked_results if r.score >= threshold]
            trace["valid_result_count"] = len(valid_results)

            # === PHASE 4: OUTPUT FORMATTING ===
            latency_ms = int((time.perf_counter() - start_time) * 1000)

            if not valid_results:
                result = SearchResult(
                    status=SearchStatus.ABSTAINED,
                    answer_context="No relevant context found in knowledge base for this query.",
                    query_used=rewritten_query,
                    metadata_applied=metadata_filters,
                    _latency_ms=latency_ms,
                )
            else:
                # Truncate context to respect agent token budget
                context = self._format_context(valid_results, max_tokens=2000)
                sources = [
                    {
                        "chunk_id": r.chunk_id,
                        "doc_title": r.doc_title,
                        "relevance_score": round(r.score, 3),
                    }
                    for r in valid_results
                ]
                result = SearchResult(
                    status=SearchStatus.FOUND,
                    answer_context=context,
                    sources=sources,
                    query_used=rewritten_query,
                    metadata_applied=metadata_filters,
                    _latency_ms=latency_ms,
                )

        except Exception as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            logger.error(f"RAG tool error: {str(e)}", exc_info=True)
            result = SearchResult(
                status=SearchStatus.ERROR,
                answer_context=f"Knowledge search failed: {type(e).__name__}",
                query_used=query,
                _latency_ms=latency_ms,
            )
            trace["error"] = str(e)

        # === OBSERVABILITY: LOG FULL TRACE ===
        trace["result_status"] = result.status.value
        trace["latency_ms"] = result._latency_ms
        logger.info(json.dumps(trace))

        return result.to_dict()

    # --- PRIVATE HELPER METHODS (Implement based on your stack) ---

    def _rewrite_query(self, query: str, thought_context: str) -> str:
        """Resolve coreferences, decompose multi-intent queries."""
        prompt = f"""Rewrite this agent query into a self-contained search statement.
        Original: {query}
        Agent Thought Context: {thought_context}
        Rules: Resolve pronouns using context. Remove conversational filler. Output ONLY the rewritten query."""
        return self.rewriter.generate(prompt, max_tokens=100).strip()

    def _extract_metadata(self, query: str) -> dict:
        """Extract structured filters from rewritten query."""
        # Use lightweight NER or small LLM call
        # Return empty dict if no filters detected
        pass

    def _rrf_fusion(
        self, vector_results: list, bm25_results: list, k: int = 60
    ) -> list:
        """Reciprocal Rank Fusion with source attribution preservation."""
        scores = {}
        for rank, r in enumerate(vector_results):
            scores[r.chunk_id] = scores.get(r.chunk_id, 0) + 1.0 / (k + rank + 1)
            scores[r.chunk_id] = {
                "score": scores[r.chunk_id],
                "source": r,
                "arm": "vector",
            }
        for rank, r in enumerate(bm25_results):
            scores[r.chunk_id] = scores.get(r.chunk_id, 0) + 1.0 / (k + rank + 1)
            if isinstance(scores[r.chunk_id], dict):
                scores[r.chunk_id]["score"] += 1.0 / (k + rank + 1)
                scores[r.chunk_id]["arm"] = "hybrid"
            else:
                scores[r.chunk_id] = {
                    "score": 1.0 / (k + rank + 1),
                    "source": r,
                    "arm": "bm25",
                }

        sorted_items = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)
        return [item[1]["source"] for item in sorted_items]

    def _compute_dynamic_threshold(self, reranked_results: list) -> float:
        """Mean + 0.5*std of reranker scores. Falls back to config default if <3 results."""
        if len(reranked_results) < 3:
            return self.config["default_abstention_threshold"]
        scores = [r.score for r in reranked_results]
        mean_score = sum(scores) / len(scores)
        std_score = (sum((s - mean_score) ** 2 for s in scores) / len(scores)) ** 0.5
        return max(mean_score + 0.5 * std_score, self.config["min_absolute_threshold"])

    def _format_context(self, results: list, max_tokens: int) -> str:
        """Concatenate chunks with source markers. Truncate at token boundary."""
        # Implement token-aware truncation (e.g., tiktoken)
        # Format: [Source: doc_title | chunk_id]\n{content}\n\n
        pass
