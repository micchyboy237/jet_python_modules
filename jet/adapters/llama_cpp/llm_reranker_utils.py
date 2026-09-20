"""LLM-based reranker for complex relevance criteria with explainability."""

from __future__ import annotations

import json
from typing import Optional, TypedDict, overload

from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.llama_cpp.factory import get_async_llm_client, get_llm_client
from jet.adapters.llama_cpp.llm_utils import achat, chat
from jet.logger import logger
from pydantic import BaseModel, Field


class RankingResultDict(TypedDict):
    """Typed dictionary for a single ranked document with explanation."""

    index: int
    score: float
    document: str
    reason: str


class CompactRankingResultDict(TypedDict):
    """Typed dictionary for minimal ranking result (token-efficient)."""

    index: int
    score: float
    document: str


class RankingResult(BaseModel):
    """Single ranked document with score and explanation."""

    index: int = Field(description="Original document index")
    score: float = Field(description="Relevance score from 0 to 10")
    document: str = Field(description="The document text")
    reason: str = Field(default="", description="Brief explanation for the ranking")


class CompactRankingResult(BaseModel):
    """Minimal ranking result for token-efficient responses."""

    index: int = Field(description="Original document index")
    score: float = Field(description="Relevance score from 0 to 10")


class RankingsResponse(BaseModel):
    """Structured response schema for LLM reranking."""

    rankings: list[RankingResult] = Field(
        description="Ranked documents ordered by relevance"
    )


class CompactRankingsResponse(BaseModel):
    """Compact response schema for LLM reranking."""

    rankings: list[CompactRankingResult] = Field(
        description="Ranked documents ordered by relevance"
    )


class LLMReranker:
    """
    Use LLM to rerank documents with explainable relevance.
    Reuses llm_utils.chat/achat for structured output validation and
    factory.py for consistent client configuration.
    """

    def __init__(
        self,
        model: str = LLM_MODEL,
        base_url: Optional[str] = None,
        api_key: Optional[str] = "not-needed",
    ):
        self.model = model
        self._base_url = base_url
        self._api_key = api_key
        logger.debug(f"LLMReranker initialized with model={model}")

    def _get_sync_client(self):
        return get_llm_client(base_url=self._base_url, api_key=self._api_key)

    def _get_async_client(self):
        return get_async_llm_client(base_url=self._base_url, api_key=self._api_key)

    @overload
    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
        min_score: float = 7.0,
        criteria: Optional[str] = None,
        max_doc_length: int = 200,
        include_reasoning: bool = False,
    ) -> list[CompactRankingResultDict]: ...

    @overload
    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
        min_score: float = 7.0,
        criteria: Optional[str] = None,
        max_doc_length: int = 200,
        include_reasoning: bool = True,
    ) -> list[RankingResultDict]: ...

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
        min_score: float = 7.0,
        criteria: Optional[str] = None,
        max_doc_length: int = 200,
        include_reasoning: bool = False,
    ) -> list[RankingResultDict | CompactRankingResultDict]:
        """Synchronous rerank using llm_utils.chat with structured output."""
        prompt = self._build_optimized_prompt(
            query, documents, criteria, max_doc_length, include_reasoning
        )

        logger.info(f"Reranking {len(documents)} docs for query='{query[:60]}...'")

        response_format = (
            RankingsResponse if include_reasoning else CompactRankingsResponse
        )

        result = chat(
            prompt_or_messages=[
                {
                    "role": "system",
                    "content": "You are a concise relevance ranking engine.",
                },
                {"role": "user", "content": prompt},
            ],
            model=self.model,
            client=self._get_sync_client(),
            temperature=0.0,
            enable_thinking=False,
            response_format=response_format,
        )

        ranked_items = self._extract_results(
            result, documents, top_k, include_reasoning
        )
        # Apply threshold filter
        return [item for item in ranked_items if item["score"] >= min_score]

    @overload
    async def arerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
        min_score: float = 7.0,
        criteria: Optional[str] = None,
        max_doc_length: int = 200,
        include_reasoning: bool = False,
    ) -> list[CompactRankingResultDict]: ...

    @overload
    async def arerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
        min_score: float = 7.0,
        criteria: Optional[str] = None,
        max_doc_length: int = 200,
        include_reasoning: bool = True,
    ) -> list[RankingResultDict]: ...

    async def arerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
        min_score: float = 7.0,
        criteria: Optional[str] = None,
        max_doc_length: int = 200,
        include_reasoning: bool = False,
    ) -> list[RankingResultDict | CompactRankingResultDict]:
        """Async rerank using llm_utils.achat with structured output."""
        prompt = self._build_optimized_prompt(
            query, documents, criteria, max_doc_length, include_reasoning
        )

        logger.info(
            f"Async reranking {len(documents)} docs for query='{query[:60]}...'"
        )

        response_format = (
            RankingsResponse if include_reasoning else CompactRankingsResponse
        )

        result = await achat(
            prompt_or_messages=[
                {
                    "role": "system",
                    "content": "You are a concise relevance ranking engine.",
                },
                {"role": "user", "content": prompt},
            ],
            model=self.model,
            client=self._get_async_client(),
            temperature=0.0,
            enable_thinking=False,
            response_format=response_format,
        )

        ranked_items = self._extract_results(
            result, documents, top_k, include_reasoning
        )
        return [item for item in ranked_items if item["score"] >= min_score]

    def _extract_results(
        self,
        result,
        documents: list[str],
        top_k: int,
        include_reasoning: bool,
    ) -> list[RankingResultDict | CompactRankingResultDict]:
        """Extract ranked results from StreamCompletionResult, with fallback."""
        if result.structured and result.structured.success:
            rankings = result.structured.parsed.rankings
            logger.debug(f"Structured parse succeeded: {len(rankings)} rankings")

            final_results: list[RankingResultDict | CompactRankingResultDict] = []
            for r in rankings[:top_k]:
                item = r.model_dump() if hasattr(r, "model_dump") else r
                # Ensure we map back to the full original document text
                if "index" in item and item["index"] < len(documents):
                    item["document"] = documents[item["index"]]
                # Guarantee 'reason' key exists when reasoning is enabled
                if include_reasoning and "reason" not in item:
                    item["reason"] = ""
                final_results.append(item)
            return final_results

        logger.warning(
            f"Structured parse failed: {result.structured.error if result.structured else 'no structured result'}. "
            "Falling back to text parsing."
        )
        return self._parse_text_response(result.content, documents, include_reasoning)[
            :top_k
        ]

    def _build_optimized_prompt(
        self,
        query: str,
        documents: list[str],
        criteria: Optional[str],
        max_doc_length: int,
        include_reasoning: bool,
    ) -> str:
        """Build a token-efficient ranking prompt."""
        # Truncate documents to reduce input tokens
        truncated_docs = [
            doc[:max_doc_length] + "..." if len(doc) > max_doc_length else doc
            for doc in documents
        ]

        docs_text = "\n".join([f"[{i}] {doc}" for i, doc in enumerate(truncated_docs)])

        criteria_text = f"\nCriteria: {criteria}" if criteria else ""
        explanation_note = ', "reason": brief explanation' if include_reasoning else ""

        return f"""Query: {query}{criteria_text}
Documents:
{docs_text}

Return a JSON object with a "rankings" array. Each ranking should have:
- "index": original document index
- "score": relevance score from 0 to 10{explanation_note}
Only return valid JSON, no other text."""

    def _parse_text_response(
        self,
        text: str,
        documents: list[str],
        include_reasoning: bool,
    ) -> list[RankingResultDict | CompactRankingResultDict]:
        """Fallback parser for non-JSON responses."""
        try:
            result = json.loads(text)
            if isinstance(result, list):
                rankings = result
            else:
                rankings = result.get("rankings", [])

            # Map back to full documents
            final_results: list[RankingResultDict | CompactRankingResultDict] = []
            for r in rankings:
                if "index" in r and r["index"] < len(documents):
                    r["document"] = documents[r["index"]]
                if include_reasoning and "reason" not in r:
                    r["reason"] = ""
                final_results.append(r)
            return final_results

        except (json.JSONDecodeError, KeyError):
            logger.error(f"Failed to parse rerank response: {text[:200]}...")
            error_result: RankingResultDict = {
                "index": 0,
                "score": 0.0,
                "document": documents[0] if documents else "Error parsing LLM response",
                "reason": f"Raw response: {text[:200]}...",
            }
            return [error_result]


if __name__ == "__main__":
    query = "What is the best programming language for data science?"
    documents = [
        "Python is widely used in data science due to libraries like pandas, numpy, and scikit-learn.",
        "JavaScript is primarily used for web development and browser-based applications.",
        "R language was specifically designed for statistical computing and data analysis.",
        "Java is used in enterprise applications and Android development.",
        "Python's machine learning ecosystem includes TensorFlow, PyTorch, and scikit-learn.",
    ]
    print("=" * 60)
    print("OPTIMIZED LLM-BASED RERANKING")
    print("=" * 60)
    print(f"Query: {query}\n")

    reranker = LLMReranker()

    # Test with reasoning enabled and threshold
    results = reranker.rerank(
        query=query,
        documents=documents,
        top_k=3,
        min_score=7.0,
        criteria="Consider ecosystem maturity and library support",
        include_reasoning=True,
    )

    if not results:
        print("No documents met the minimum score threshold.")
    else:
        for i, r in enumerate(results, 1):
            print(f"{i}. [score:{r['score']}/10] {r['document'][:100]}...")
            if "reason" in r and r["reason"]:
                print(f"   Reason: {r['reason']}")
            print()
