"""LLM-based reranker for complex relevance criteria with explainability."""

from __future__ import annotations

import json
from typing import Optional

from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.llama_cpp.factory import get_async_llm_client, get_llm_client
from jet.adapters.llama_cpp.llm_utils import achat, chat
from jet.logger import logger
from pydantic import BaseModel, Field


class RankingResult(BaseModel):
    """Single ranked document with score and explanation."""

    index: int = Field(description="Original document index")
    score: float = Field(description="Relevance score from 0 to 10")
    document: str = Field(description="The document text")
    reason: str = Field(default="", description="Brief explanation for the ranking")


class RankingsResponse(BaseModel):
    """Structured response schema for LLM reranking."""

    rankings: list[RankingResult] = Field(
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
        # Store params; actual clients created lazily via factory to respect overrides
        self.model = model
        self._base_url = base_url
        self._api_key = api_key
        logger.debug(f"LLMReranker initialized with model={model}")

    def _get_sync_client(self):
        return get_llm_client(base_url=self._base_url, api_key=self._api_key)

    def _get_async_client(self):
        return get_async_llm_client(base_url=self._base_url, api_key=self._api_key)

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
        criteria: Optional[str] = None,
        return_explanation: bool = True,
    ) -> list[dict]:
        """Synchronous rerank using llm_utils.chat with structured output."""
        prompt = self._build_prompt(query, documents, criteria, return_explanation)
        logger.info(f"Reranking {len(documents)} docs for query='{query[:60]}...'")

        result = chat(
            prompt_or_messages=[
                {"role": "system", "content": "You are a relevance ranking expert."},
                {"role": "user", "content": prompt},
            ],
            model=self.model,
            client=self._get_sync_client(),
            temperature=0.0,
            enable_thinking=False,
            response_format=RankingsResponse,
        )

        return self._extract_results(result, documents, top_k)

    async def arerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
        criteria: Optional[str] = None,
        return_explanation: bool = True,
    ) -> list[dict]:
        """Async rerank using llm_utils.achat with structured output."""
        prompt = self._build_prompt(query, documents, criteria, return_explanation)
        logger.info(
            f"Async reranking {len(documents)} docs for query='{query[:60]}...'"
        )

        result = await achat(
            prompt_or_messages=[
                {"role": "system", "content": "You are a relevance ranking expert."},
                {"role": "user", "content": prompt},
            ],
            model=self.model,
            client=self._get_async_client(),
            temperature=0.0,
            enable_thinking=False,
            response_format=RankingsResponse,
        )

        return self._extract_results(result, documents, top_k)

    def _extract_results(self, result, documents: list[str], top_k: int) -> list[dict]:
        """Extract ranked results from StreamCompletionResult, with fallback."""
        if result.structured and result.structured.success:
            rankings = result.structured.parsed.rankings
            logger.debug(f"Structured parse succeeded: {len(rankings)} rankings")
            return [r.model_dump() for r in rankings[:top_k]]

        # Fallback to raw content parsing
        logger.warning(
            f"Structured parse failed: {result.structured.error if result.structured else 'no structured result'}. "
            "Falling back to text parsing."
        )
        return self._parse_text_response(result.content, documents)[:top_k]

    def _build_prompt(
        self,
        query: str,
        documents: list[str],
        criteria: Optional[str],
        return_explanation: bool,
    ) -> str:
        """Build the ranking prompt."""
        docs_text = "\n\n".join(
            [f"Document {i}:\n{doc}" for i, doc in enumerate(documents)]
        )
        criteria_text = (
            f"\n\nAdditional relevance criteria:\n{criteria}" if criteria else ""
        )
        explanation_note = ', "reason": brief explanation' if return_explanation else ""
        return f"""Rank the following documents by their relevance to the query.
Query: {query}{criteria_text}
Documents:
{docs_text}
Return a JSON object with a "rankings" array. Each ranking should have:
- "index": original document index
- "score": relevance score from 0 to 10
- "document": the document text{explanation_note}
Only return valid JSON, no other text."""

    def _parse_text_response(self, text: str, documents: list[str]) -> list[dict]:
        """Fallback parser for non-JSON responses."""
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
            return result.get("rankings", [])
        except (json.JSONDecodeError, KeyError):
            logger.error(f"Failed to parse rerank response: {text[:200]}...")
            return [
                {
                    "index": 0,
                    "score": 0.0,
                    "document": "Error parsing LLM response",
                    "reason": f"Raw response: {text[:200]}...",
                }
            ]


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
    print("LLM-BASED RERANKING")
    print("=" * 60)
    print(f"Query: {query}\n")
    reranker = LLMReranker()
    results = reranker.rerank(
        query=query,
        documents=documents,
        top_k=3,
        criteria="Consider ecosystem maturity, learning curve, and community support",
        return_explanation=True,
    )
    for i, r in enumerate(results, 1):
        print(f"{i}. [score:{r['score']}/10] {r['document'][:100]}...")
        if "reason" in r:
            print(f"   Reason: {r['reason']}")
        print()
