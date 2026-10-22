"""Standalone LLM Reranker Utilities (Functional API).
Delegates all logic to LLMReranker class for consistency and DRY compliance.
Provides synchronous and asynchronous reranking with metadata support.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.llama_cpp.llm_reranker_utils_class import LLMReranker
from jet.adapters.llama_cpp.llm_reranker_utils_types import (
    CompactRankingResult,
    RankingResult,
)


def rerank(
    query: str,
    documents: list[str],
    metadata: list[dict[str, Any]] | None = None,
    metadata_format_fn: Callable[[dict[str, Any]], str] | None = None,
    top_k: int | None = None,
    min_score: float = 6.0,
    criteria: str | None = None,
    max_doc_tokens: int | None = None,
    include_reasoning: bool = False,
    model: str = LLM_MODEL,
    base_url: str | None = None,
    api_key: str | None = "not-needed",
    max_tokens: int | None = None,
    batch_overlap: int = 0,
    min_results: int | None = None,
    min_high_results: int | None = None,
    high_score_threshold: float = 9.0,
) -> list[RankingResult] | list[CompactRankingResult]:
    """
    Synchronous functional interface for LLM-based reranking.
    Args:
        query: Search query string.
        documents: List of document strings to rerank.
        metadata: Optional list of metadata dicts corresponding to each document.
        metadata_format_fn: Function to convert metadata dict to string for context injection.
        top_k: Number of top results to return. If None, returns all results that meet min_score.
        min_score: Minimum relevance score threshold.
            With Tiered Scoring:
            - > 6.0: Only "High" relevance (Score 9.0)
            - <= 6.0: "High" and "Medium" relevance (Scores 9.0, 6.0)
            - <= 3.0: All tiers (Scores 9.0, 6.0, 3.0)
        criteria: Optional specific criteria for the LLM to focus on.
        max_doc_tokens: Max tokens per document after truncation.
        include_reasoning: If True, LLM provides a reason for each ranking.
        model: LLM model to use for ranking.
        base_url: Base URL for the LLM server.
        api_key: API key for the LLM server.
        max_tokens: Max context window tokens.
        batch_overlap: Number of documents to overlap between batches.
        min_results: If set, stop processing batches once this many total results are found.
        min_high_results: If set, stop processing batches once this many 'high' quality results are found.
        high_score_threshold: Score threshold to define a 'high' quality result (default 9.0).
    Returns:
        List of RankingResult (if include_reasoning=True) or CompactRankingResult.
    """
    reranker = LLMReranker(
        model=model,
        base_url=base_url,
        api_key=api_key,
        max_tokens=max_tokens,
        batch_overlap=batch_overlap,
    )
    return reranker.rerank(
        query=query,
        documents=documents,
        metadata=metadata,
        metadata_format_fn=metadata_format_fn,
        top_k=top_k,
        min_score=min_score,
        criteria=criteria,
        max_doc_tokens=max_doc_tokens,
        include_reasoning=include_reasoning,
        min_results=min_results,
        min_high_results=min_high_results,
        high_score_threshold=high_score_threshold,
    )


async def arerank(
    query: str,
    documents: list[str],
    metadata: list[dict[str, Any]] | None = None,
    metadata_format_fn: Callable[[dict[str, Any]], str] | None = None,
    top_k: int | None = None,
    min_score: float = 6.0,
    criteria: str | None = None,
    max_doc_tokens: int | None = None,
    include_reasoning: bool = False,
    model: str = LLM_MODEL,
    base_url: str | None = None,
    api_key: str | None = "not-needed",
    max_tokens: int | None = None,
    batch_overlap: int = 0,
    min_results: int | None = None,
    min_high_results: int | None = None,
    high_score_threshold: float = 9.0,
) -> list[RankingResult] | list[CompactRankingResult]:
    """
    Asynchronous functional interface for LLM-based reranking.
    Args:
        query: Search query string.
        documents: List of document strings to rerank.
        metadata: Optional list of metadata dicts corresponding to each document.
        metadata_format_fn: Function to convert metadata dict to string for context injection.
        top_k: Number of top results to return. If None, returns all results that meet min_score.
        min_score: Minimum relevance score threshold.
        criteria: Optional specific criteria for the LLM to focus on.
        max_doc_tokens: Max tokens per document after truncation.
        include_reasoning: If True, LLM provides a reason for each ranking.
        model: LLM model to use for ranking.
        base_url: Base URL for the LLM server.
        api_key: API key for the LLM server.
        max_tokens: Max context window tokens.
        batch_overlap: Number of documents to overlap between batches.
        min_results: If set, stop processing batches once this many total results are found.
        min_high_results: If set, stop processing batches once this many 'high' quality results are found.
        high_score_threshold: Score threshold to define a 'high' quality result (default 9.0).
    Returns:
        List of RankingResult (if include_reasoning=True) or CompactRankingResult.
    """
    reranker = LLMReranker(
        model=model,
        base_url=base_url,
        api_key=api_key,
        max_tokens=max_tokens,
        batch_overlap=batch_overlap,
    )
    return await reranker.arerank(
        query=query,
        documents=documents,
        metadata=metadata,
        metadata_format_fn=metadata_format_fn,
        top_k=top_k,
        min_score=min_score,
        criteria=criteria,
        max_doc_tokens=max_doc_tokens,
        include_reasoning=include_reasoning,
        min_results=min_results,
        min_high_results=min_high_results,
        high_score_threshold=high_score_threshold,
    )


if __name__ == "__main__":
    import asyncio

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
    print("FUNCTIONAL API RERANKING EVALUATION")
    print("=" * 60)
    print(f"Query: {query}\n")

    print("--- TEST 1: Sync Rerank with Metadata Context ---")
    results_sync = rerank(
        query=query,
        documents=documents,
        metadata=meta,
        top_k=3,
        min_score=6.0,
        include_reasoning=False,
    )
    for i, r in enumerate(results_sync, 1):
        print(f"{i}. [score:{r['score']}/10] Index: {r['index']} Meta: {r['metadata']}")

    print("\n--- TEST 2: Async Rerank with Reasoning ---")

    async def run_async_test():
        results_async = await arerank(
            query=query,
            documents=documents,
            metadata=meta,
            top_k=3,
            min_score=6.0,
            criteria="Consider library maturity",
            include_reasoning=True,
        )
        for i, r in enumerate(results_async, 1):
            print(
                f"{i}. [score:{r['score']}/10] Index: {r['index']} Meta: {r['metadata']}"
            )
            if "reason" in r:
                print(f"   Reason: {r['reason']}")

    asyncio.run(run_async_test())

    print("\n" + "=" * 60)
    print("Evaluation Complete")
    print("=" * 60)
