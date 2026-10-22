"""LLM-based reranker using GBNF Grammar with Semantic Anchors & Token-Aware Truncation."""

from __future__ import annotations

import json
from typing import Any, Callable, List, Optional, Tuple, overload

from jet.adapters.llama_cpp.chunking_utils.truncation import truncate_texts
from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.llama_cpp.factory import get_async_llm_client, get_llm_client
from jet.adapters.llama_cpp.llm_reranker_utils_types import (
    CompactRankingResult,
    RankingResult,
)
from jet.adapters.llama_cpp.llm_utils_observed import achat, chat
from jet.adapters.llama_cpp.model_utils import get_model_ctx_embd_size
from jet.adapters.llama_cpp.token_utils import count_tokens
from jet.logger import logger

COMPACT_GRAMMAR = r"""
root ::= rankings
rankings ::= "[" item ("," item)* "]"
item ::= "{" ws "\"index\"" ws ":" ws int ws "," ws "\"score\"" ws ":" ws number ws "}"
ws ::= [ \t\n]*
int ::= [0-9]+
number ::= [0-9]+ "." [0-9]+
"""

FULL_GRAMMAR = r"""
root ::= rankings
rankings ::= "[" item ("," item)* "]"
item ::= "{" ws "\"index\"" ws ":" ws int ws "," ws "\"score\"" ws ":" ws number ws "," ws "\"reason\"" ws ":" ws string ws "}"
ws ::= [ \t\n]*
int ::= [0-9]+
number ::= [0-9]+ "." [0-9]+
string ::= "\"" char* "\""
char ::= [^"\\] | "\\" ["\\/bfnrt]
"""


class BatchedTokenProcessor:
    """
    Splits documents into batches based on dynamic token limits.
    Enhanced to support batch overlap for cross-batch continuity.
    """

    def __init__(self, model: str = LLM_MODEL):
        self.model = model
        logger.debug(f"BatchedTokenProcessor initialized with model={model}")

    def create_batches(
        self,
        documents: List[str],
        max_tokens: int = 2048,
        query: Optional[str] = None,
        anchor_reserve: int = 300,
        batch_overlap: int = 0,
    ) -> List[Tuple[List[str], List[int]]]:
        if not documents:
            return []

        query_tokens = count_tokens(query, model=self.model) if query else 0
        overhead = 200 + anchor_reserve
        available_budget = max_tokens - query_tokens - overhead

        if available_budget <= 0:
            raise ValueError(
                f"max_tokens ({max_tokens}) is too low to accommodate query "
                f"({query_tokens}) and overhead ({overhead})."
            )

        batches = []
        current_batch_docs = []
        current_batch_indices = []
        current_batch_tokens = 0

        for i, doc in enumerate(documents):
            doc_tokens = count_tokens(doc, model=self.model)
            if doc_tokens > available_budget:
                logger.warning(
                    f"Document[{i}] ({doc_tokens} tokens) exceeds batch budget "
                    f"({available_budget}). Processing in isolated batch."
                )
                if current_batch_docs:
                    batches.append((current_batch_docs, current_batch_indices))
                    current_batch_docs, current_batch_indices, current_batch_tokens = (
                        [],
                        [],
                        0,
                    )
                batches.append(([doc], [i]))
                continue

            if current_batch_tokens + doc_tokens > available_budget:
                if current_batch_docs:
                    batches.append((current_batch_docs, current_batch_indices))

                overlap_docs = []
                overlap_indices = []
                overlap_tokens = 0

                if batch_overlap > 0 and current_batch_docs:
                    for od, oi in zip(
                        reversed(current_batch_docs), reversed(current_batch_indices)
                    ):
                        ot = count_tokens(od, model=self.model)
                        if overlap_tokens + ot > available_budget // 4:
                            break
                        overlap_docs.insert(0, od)
                        overlap_indices.insert(0, oi)
                        overlap_tokens += ot

                    if overlap_docs:
                        logger.debug(
                            f"Carrying forward {len(overlap_docs)} overlap docs to next batch"
                        )

                current_batch_docs = overlap_docs
                current_batch_indices = overlap_indices
                current_batch_tokens = overlap_tokens

            current_batch_docs.append(doc)
            current_batch_indices.append(i)
            current_batch_tokens += doc_tokens

        if current_batch_docs:
            batches.append((current_batch_docs, current_batch_indices))

        logger.info(
            f"Created {len(batches)} batches with budget={available_budget}, overlap={batch_overlap}"
        )
        return batches


class LLMReranker:
    """
    Use LLM to rerank documents using GBNF Grammar with Semantic Anchors.
    Enhancements over baseline:
    1. Token-aware sentence truncation (no mid-sentence cuts)
    2. Dynamic context window detection
    3. Semantic anchor calibration across batches
    4. Optional batch overlap for cross-boundary comparison
    5. Metadata injection into ranking context
    """

    def __init__(
        self,
        model: str = LLM_MODEL,
        base_url: Optional[str] = None,
        api_key: Optional[str] = "not-needed",
        max_tokens: Optional[int] = None,
        batch_overlap: int = 0,
    ):
        self.model = model
        self._base_url = base_url
        self._api_key = api_key
        self.batch_overlap = batch_overlap

        if max_tokens is None:
            try:
                ctx_info = get_model_ctx_embd_size(model)
                self.max_tokens = int(ctx_info.get("ctx", 4096) * 0.75)
                logger.info(
                    f"Auto-detected context for {model}: {ctx_info.get('ctx')} → using {self.max_tokens} for reranking"
                )
            except Exception as e:
                logger.warning(
                    f"Could not detect context size ({e}), defaulting to 2048"
                )
                self.max_tokens = 2048
        else:
            self.max_tokens = max_tokens

        self._processor = BatchedTokenProcessor(model=model)
        logger.debug(
            f"LLMReranker initialized: model={model}, max_tokens={self.max_tokens}, overlap={batch_overlap}"
        )

    def _get_sync_client(self):
        return get_llm_client(base_url=self._base_url, api_key=self._api_key)

    def _get_async_client(self):
        return get_async_llm_client(base_url=self._base_url, api_key=self._api_key)

    @staticmethod
    def _validate_metadata(
        documents: list[str], metadata: list[dict[str, Any]] | None
    ) -> list[dict[str, Any]] | None:
        """Validate that metadata length matches documents length if provided."""
        if metadata is None:
            return None
        if len(metadata) != len(documents):
            raise ValueError(
                f"Length of metadata ({len(metadata)}) must match length of documents ({len(documents)})"
            )
        return metadata

    @staticmethod
    def _default_metadata_format(meta: dict[str, Any]) -> str:
        """
        Default formatter that extracts common fields into a concise header.
        Example output: "[Source: wiki | Date: 2023]"
        """
        if not meta:
            return ""
        parts = []
        key_map = {
            "source": "Source",
            "title": "Title",
            "date": "Date",
            "year": "Year",
            "author": "Author",
            "url": "URL",
        }
        for key, label in key_map.items():
            if key in meta and meta[key]:
                val = str(meta[key])
                if len(val) > 50:
                    val = val[:47] + "..."
                parts.append(f"{label}: {val}")

        if not parts:
            parts = [f"{k}: {v}" for k, v in meta.items() if v]

        return f"[{' | '.join(parts)}]" if parts else ""

    @staticmethod
    def _inject_metadata_context(
        documents: list[str],
        metadata: list[dict[str, Any]] | None,
        format_fn: Callable[[dict[str, Any]], str] | None,
    ) -> list[str]:
        """
        Prepend metadata to documents. Uses _default_metadata_format if no fn provided.
        This allows the ranking model to 'see' the metadata during evaluation.
        """
        if not metadata:
            return documents

        active_fn = format_fn or LLMReranker._default_metadata_format
        enhanced_docs = []
        for doc, meta in zip(documents, metadata):
            try:
                meta_str = active_fn(meta)
                if meta_str:
                    enhanced_docs.append(f"{meta_str}\n{doc}")
                else:
                    enhanced_docs.append(doc)
            except Exception as e:
                logger.warning(f"Failed to format metadata for doc: {e}")
                enhanced_docs.append(doc)

        return enhanced_docs

    def _truncate_semantically(self, texts: List[str], max_tokens: int) -> List[str]:
        """
        Truncate texts to token limit while preserving sentence boundaries.
        Uses existing truncate_texts utility with strict_sentences=True.
        """
        if not texts:
            return []
        try:
            return truncate_texts(
                texts=texts,
                model=self.model,
                max_tokens=max_tokens,
                strict_sentences=True,
                show_progress=False,
            )
        except Exception as e:
            logger.warning(
                f"Semantic truncation failed ({e}), falling back to character truncation"
            )
            approx_chars = max_tokens * 4
            return [
                t[:approx_chars] + "..." if len(t) > approx_chars else t for t in texts
            ]

    def _get_anchors(
        self,
        previous_results: List[CompactRankingResult],
        original_documents: List[str],
        max_anchor_tokens: int = 80,
    ) -> List[str]:
        """
        Extract Top, Median, and Bottom documents as semantically-truncated anchors.
        Each anchor is individually truncated to max_anchor_tokens to preserve
        sentence structure within the limited anchor budget.
        """
        if not previous_results:
            return []

        sorted_results = sorted(
            previous_results, key=lambda x: x["score"], reverse=True
        )
        indices_to_fetch = set()
        indices_to_fetch.add(sorted_results[0]["index"])
        mid_idx = len(sorted_results) // 2
        indices_to_fetch.add(sorted_results[mid_idx]["index"])
        indices_to_fetch.add(sorted_results[-1]["index"])

        raw_anchors = []
        for idx in indices_to_fetch:
            if idx < len(original_documents):
                score = next(
                    (r["score"] for r in previous_results if r["index"] == idx), 0
                )
                raw_anchors.append((idx, original_documents[idx], score))

        anchor_texts = [a[1] for a in raw_anchors]
        truncated = self._truncate_semantically(anchor_texts, max_anchor_tokens)

        result = []
        for (idx, _, score), trunc_text in zip(raw_anchors, truncated):
            result.append(f"[ANCHOR ID: {idx}] [Score: {score:.1f}/10] {trunc_text}")

        return result

    @overload
    def rerank(
        self,
        query: str,
        documents: list[str],
        metadata: list[dict[str, Any]] | None = None,
        metadata_format_fn: Callable[[dict[str, Any]], str] | None = None,
        top_k: int | None = None,
        min_score: float = 7.0,
        criteria: Optional[str] = None,
        max_doc_tokens: int | None = None,
        include_reasoning: bool = False,
        max_tokens: Optional[int] = None,
    ) -> list[CompactRankingResult]: ...

    @overload
    def rerank(
        self,
        query: str,
        documents: list[str],
        metadata: list[dict[str, Any]] | None = None,
        metadata_format_fn: Callable[[dict[str, Any]], str] | None = None,
        top_k: int | None = None,
        min_score: float = 7.0,
        criteria: Optional[str] = None,
        max_doc_tokens: int | None = None,
        include_reasoning: bool = True,
        max_tokens: Optional[int] = None,
    ) -> list[RankingResult]: ...

    def rerank(
        self,
        query: str,
        documents: list[str],
        metadata: list[dict[str, Any]] | None = None,
        metadata_format_fn: Callable[[dict[str, Any]], str] | None = None,
        top_k: int | None = None,
        min_score: float = 7.0,
        criteria: Optional[str] = None,
        max_doc_tokens: int | None = None,
        include_reasoning: bool = False,
        max_tokens: Optional[int] = None,
    ) -> list[RankingResult | CompactRankingResult]:
        """Synchronous rerank with semantic truncation and anchor calibration."""
        valid_metadata = self._validate_metadata(documents, metadata)
        docs_for_ranking = self._inject_metadata_context(
            documents, valid_metadata, metadata_format_fn
        )

        limit = max_tokens if max_tokens is not None else self.max_tokens
        batches = self._processor.create_batches(
            documents=docs_for_ranking,
            max_tokens=limit,
            query=query,
            anchor_reserve=(max_doc_tokens * 3 + 100) if max_doc_tokens else 300,
            batch_overlap=self.batch_overlap,
        )

        all_results: List[CompactRankingResult] = []
        anchor_docs: List[str] = []

        for batch_idx, (batch_docs, original_indices) in enumerate(batches):
            logger.info(
                f"Processing batch {batch_idx + 1}/{len(batches)} ({len(batch_docs)} docs)"
            )

            # Handle optional max_doc_tokens
            if max_doc_tokens is not None:
                truncated_docs = self._truncate_semantically(batch_docs, max_doc_tokens)
            else:
                truncated_docs = batch_docs

            prompt = self._build_grammar_prompt(
                query,
                truncated_docs,
                criteria,
                top_k,
                min_score,
                include_reasoning,
                anchor_docs=anchor_docs,
            )

            grammar_str = FULL_GRAMMAR if include_reasoning else COMPACT_GRAMMAR
            response_format = {"type": "grammar", "grammar": grammar_str}

            result = chat(
                prompt_or_messages=[
                    {
                        "role": "system",
                        "content": "You are a strict relevance ranking engine.",
                    },
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
                client=self._get_sync_client(),
                temperature=0.0,
                enable_thinking=False,
                response_format=response_format,
            )

            batch_results = self._parse_grammar_response(
                result.content, include_reasoning, min_score
            )

            seen_global_indices = {r["index"] for r in all_results}
            for res in batch_results:
                local_idx = res["index"]
                if local_idx < len(original_indices):
                    global_idx = original_indices[local_idx]
                    if global_idx not in seen_global_indices:
                        res_copy = dict(res)
                        res_copy["index"] = global_idx
                        res_copy["metadata"] = (
                            valid_metadata[global_idx] if valid_metadata else {}
                        )
                        res_copy["text"] = documents[global_idx]
                        all_results.append(res_copy)
                        seen_global_indices.add(global_idx)

            if batch_idx < len(batches) - 1:
                anchor_docs = self._get_anchors(
                    all_results,
                    docs_for_ranking,
                    max_anchor_tokens=max_doc_tokens if max_doc_tokens else 80,
                )
                if anchor_docs:
                    logger.debug(
                        f"Generated {len(anchor_docs)} semantic anchors for next batch"
                    )

        all_results.sort(key=lambda x: x["score"], reverse=True)

        # Handle optional top_k
        if top_k is None:
            return all_results
        return all_results[:top_k]

    @overload
    async def arerank(
        self,
        query: str,
        documents: list[str],
        metadata: list[dict[str, Any]] | None = None,
        metadata_format_fn: Callable[[dict[str, Any]], str] | None = None,
        top_k: int | None = None,
        min_score: float = 7.0,
        criteria: Optional[str] = None,
        max_doc_tokens: int | None = None,
        include_reasoning: bool = False,
        max_tokens: Optional[int] = None,
    ) -> list[CompactRankingResult]: ...

    @overload
    async def arerank(
        self,
        query: str,
        documents: list[str],
        metadata: list[dict[str, Any]] | None = None,
        metadata_format_fn: Callable[[dict[str, Any]], str] | None = None,
        top_k: int | None = None,
        min_score: float = 7.0,
        criteria: Optional[str] = None,
        max_doc_tokens: int | None = None,
        include_reasoning: bool = True,
        max_tokens: Optional[int] = None,
    ) -> list[RankingResult]: ...

    async def arerank(
        self,
        query: str,
        documents: list[str],
        metadata: list[dict[str, Any]] | None = None,
        metadata_format_fn: Callable[[dict[str, Any]], str] | None = None,
        top_k: int | None = None,
        min_score: float = 7.0,
        criteria: Optional[str] = None,
        max_doc_tokens: int | None = None,
        include_reasoning: bool = False,
        max_tokens: Optional[int] = None,
    ) -> list[RankingResult | CompactRankingResult]:
        """Async rerank with semantic truncation and anchor calibration."""
        valid_metadata = self._validate_metadata(documents, metadata)
        docs_for_ranking = self._inject_metadata_context(
            documents, valid_metadata, metadata_format_fn
        )

        limit = max_tokens if max_tokens is not None else self.max_tokens
        batches = self._processor.create_batches(
            documents=docs_for_ranking,
            max_tokens=limit,
            query=query,
            anchor_reserve=(max_doc_tokens * 3 + 100) if max_doc_tokens else 300,
            batch_overlap=self.batch_overlap,
        )

        all_results: List[CompactRankingResult] = []
        anchor_docs: List[str] = []

        for batch_idx, (batch_docs, original_indices) in enumerate(batches):
            logger.info(f"Async processing batch {batch_idx + 1}/{len(batches)}")

            # Handle optional max_doc_tokens
            if max_doc_tokens is not None:
                truncated_docs = self._truncate_semantically(batch_docs, max_doc_tokens)
            else:
                truncated_docs = batch_docs

            prompt = self._build_grammar_prompt(
                query,
                truncated_docs,
                criteria,
                top_k,
                min_score,
                include_reasoning,
                anchor_docs=anchor_docs,
            )

            grammar_str = FULL_GRAMMAR if include_reasoning else COMPACT_GRAMMAR
            response_format = {"type": "grammar", "grammar": grammar_str}

            result = await achat(
                prompt_or_messages=[
                    {
                        "role": "system",
                        "content": "You are a strict relevance ranking engine.",
                    },
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
                client=self._get_async_client(),
                temperature=0.0,
                enable_thinking=False,
                response_format=response_format,
            )

            batch_results = self._parse_grammar_response(
                result.content, include_reasoning, min_score
            )

            seen_global_indices = {r["index"] for r in all_results}
            for res in batch_results:
                local_idx = res["index"]
                if local_idx < len(original_indices):
                    global_idx = original_indices[local_idx]
                    if global_idx not in seen_global_indices:
                        res_copy = dict(res)
                        res_copy["index"] = global_idx
                        res_copy["metadata"] = (
                            valid_metadata[global_idx] if valid_metadata else {}
                        )
                        res_copy["text"] = documents[global_idx]
                        all_results.append(res_copy)
                        seen_global_indices.add(global_idx)

            if batch_idx < len(batches) - 1:
                anchor_docs = self._get_anchors(
                    all_results,
                    docs_for_ranking,
                    max_anchor_tokens=max_doc_tokens if max_doc_tokens else 80,
                )

        all_results.sort(key=lambda x: x["score"], reverse=True)

        # Handle optional top_k
        if top_k is None:
            return all_results
        return all_results[:top_k]

    def _build_grammar_prompt(
        self,
        query: str,
        documents: list[str],
        criteria: Optional[str],
        top_k: int | None,
        min_score: float,
        include_reasoning: bool,
        anchor_docs: Optional[List[str]] = None,
    ) -> str:
        """Build prompt with semantic anchors and explicit scoring constraints."""
        docs_text = "\n".join([f"[ID: {i}] {doc}" for i, doc in enumerate(documents)])

        anchor_text = ""
        if anchor_docs:
            anchor_list = "\n".join(anchor_docs)
            anchor_text = f"""
REFERENCE ANCHORS (Calibrate your scoring against these):
{anchor_list}
- Significantly better than Top Anchor → score 9-10
- Similar quality to Median Anchor → score near its reference
- Clearly worse than Low Anchor → score below its reference
"""

        criteria_text = f"\nCriteria: {criteria}" if criteria else ""
        reasoning_instr = (
            " Provide a short, concise reason (max 1 sentence)."
            if include_reasoning
            else ""
        )

        top_k_instr = (
            f"Return ONLY the top {top_k} documents"
            if top_k is not None
            else "Return ALL documents"
        )

        return f"""Query: {query}{criteria_text}
{anchor_text}
Documents to Rank:
{docs_text}
Instructions:
1. Rank documents by relevance to the query.
2. Use Reference Anchors to maintain consistent scoring across batches.
3. {top_k_instr} with relevance score >= {min_score}.
4. Output MUST be a valid JSON array.{reasoning_instr}
Format:
[{{"index": ID, "score": 0-10{', "reason": "text"' if include_reasoning else ""}}}, ...]"""

    def _parse_grammar_response(
        self,
        content: str,
        include_reasoning: bool,
        min_score: float,
    ) -> list[RankingResult | CompactRankingResult]:
        """Parse the grammar-constrained JSON response."""
        try:
            data = json.loads(content)
            if not isinstance(data, list):
                data = data.get("rankings", [])

            results = []
            for item in data:
                item["index"] = int(item["index"])
                item["score"] = float(item["score"])
                if item["score"] >= min_score:
                    if include_reasoning and "reason" not in item:
                        item["reason"] = ""
                    item["metadata"] = item.get("metadata", {})
                    item["text"] = item.get("text", "")
                    results.append(item)
            return results
        except Exception as e:
            logger.error(
                f"Failed to parse grammar response: {e}. Content: {content[:200]}"
            )
            return []


if __name__ == "__main__":
    query = "What is the best programming language for data science?"
    documents = [
        "Python is widely used in data science due to libraries like pandas, numpy, and scikit-learn.",
        "JavaScript is primarily used for web development and browser-based applications.",
        "R language was specifically designed for statistical computing and data analysis.",
        "Java is used in enterprise applications and Android development.",
        "Python's machine learning ecosystem includes TensorFlow, PyTorch, and scikit-learn.",
    ]
    meta = [
        {"source": "blog", "year": 2023},
        {"source": "docs", "year": 2024},
        {"source": "wiki", "year": 2022},
        {"source": "forum", "year": 2021},
        {"source": "blog", "year": 2025},
    ]

    print("=" * 60)
    print("GRAMMAR-BASED LLM RERANKING EVALUATION")
    print("=" * 60)
    print(f"Query: {query}\n")

    reranker = LLMReranker()

    print("--- TEST 1: include_reasoning=False ---")
    results_compact = reranker.rerank(
        query=query,
        documents=documents,
        metadata=meta,
        top_k=None,
        min_score=7.0,
        include_reasoning=False,
    )
    if not results_compact:
        print("No documents met the minimum score threshold.")
    else:
        for i, r in enumerate(results_compact, 1):
            print(
                f"{i}. [score:{r['score']}/10] Index: {r['index']} Meta: {r['metadata']}"
            )

    print()
    print("--- TEST 2: include_reasoning=True ---")
    results_full = reranker.rerank(
        query=query,
        documents=documents,
        metadata=meta,
        top_k=None,
        min_score=7.0,
        criteria="Consider ecosystem maturity and library support",
        include_reasoning=True,
    )
    if not results_full:
        print("No documents met the minimum score threshold.")
    else:
        for i, r in enumerate(results_full, 1):
            print(
                f"{i}. [score:{r['score']}/10] Index: {r['index']} Meta: {r['metadata']}"
            )
            if "reason" in r and r["reason"]:
                print(f"   Reason: {r['reason']}")

    print()
    print("=" * 60)
    print("Evaluation Complete")
    print("=" * 60)
