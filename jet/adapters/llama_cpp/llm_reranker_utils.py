"""LLM-based reranker using GBNF Grammar for maximum efficiency and accuracy."""

from __future__ import annotations

import json
from typing import Optional, TypedDict, overload

from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.llama_cpp.factory import get_async_llm_client, get_llm_client
from jet.adapters.llama_cpp.llm_utils import achat, chat
from jet.logger import logger


class RankingResultDict(TypedDict):
    """Typed dictionary for a single ranked document with explanation."""

    index: int
    score: float
    reason: str


class CompactRankingResultDict(TypedDict):
    """Typed dictionary for minimal ranking result (token-efficient)."""

    index: int
    score: float


# GBNF Grammar for Compact Output (Index + Score)
COMPACT_GRAMMAR = r"""
root ::= rankings
rankings ::= "[" item ("," item)* "]"
item ::= "{" ws "\"index\"" ws ":" ws int ws "," ws "\"score\"" ws ":" ws number ws "}"
ws ::= [ \t\n]*
int ::= [0-9]+
number ::= [0-9]+ "." [0-9]+
"""

# GBNF Grammar for Full Output (Index + Score + Reason)
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


class LLMReranker:
    """
    Use LLM to rerank documents using GBNF Grammar for strict output control.
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
        """Synchronous rerank using GBNF grammar."""
        prompt = self._build_grammar_prompt(
            query,
            documents,
            criteria,
            max_doc_length,
            top_k,
            min_score,
            include_reasoning,
        )

        grammar_str = FULL_GRAMMAR if include_reasoning else COMPACT_GRAMMAR

        # We use response_format to pass the grammar to llama.cpp via extra_body
        response_format = {"type": "grammar", "grammar": grammar_str}

        logger.info(f"Reranking {len(documents)} docs for query='{query[:60]}...'")

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
            enable_thinking=False,  # Must be false for grammar
            response_format=response_format,
        )

        return self._parse_grammar_response(
            result.content, include_reasoning, min_score
        )

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
        """Async rerank using GBNF grammar."""
        prompt = self._build_grammar_prompt(
            query,
            documents,
            criteria,
            max_doc_length,
            top_k,
            min_score,
            include_reasoning,
        )

        grammar_str = FULL_GRAMMAR if include_reasoning else COMPACT_GRAMMAR
        response_format = {"type": "grammar", "grammar": grammar_str}

        logger.info(
            f"Async reranking {len(documents)} docs for query='{query[:60]}...'"
        )

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

        return self._parse_grammar_response(
            result.content, include_reasoning, min_score
        )

    def _build_grammar_prompt(
        self,
        query: str,
        documents: list[str],
        criteria: Optional[str],
        max_doc_length: int,
        top_k: int,
        min_score: float,
        include_reasoning: bool,
    ) -> str:
        """Build prompt with metadata prefixes and explicit constraints."""
        truncated_docs = [
            doc[:max_doc_length] + "..." if len(doc) > max_doc_length else doc
            for doc in documents
        ]

        # Add metadata prefix [ID: i] to help LLM track indices
        docs_text = "\n".join(
            [f"[ID: {i}] {doc}" for i, doc in enumerate(truncated_docs)]
        )

        criteria_text = f"\nCriteria: {criteria}" if criteria else ""
        reasoning_instr = (
            " Provide a short, concise reason (max 1 sentence)."
            if include_reasoning
            else ""
        )

        return f"""Query: {query}{criteria_text}

Documents:
{docs_text}

Instructions:
1. Rank the documents by relevance to the query.
2. Return ONLY the top {top_k} documents that have a relevance score of {min_score} or higher.
3. If fewer than {top_k} documents meet the threshold, return only those that do.
4. Output MUST be a valid JSON array.{reasoning_instr}

Format:
[{{"index": ID, "score": 0-10{', "reason": "text"' if include_reasoning else ""}}}, ...]"""

    def _parse_grammar_response(
        self,
        content: str,
        include_reasoning: bool,
        min_score: float,
    ) -> list[RankingResultDict | CompactRankingResultDict]:
        """Parse the grammar-constrained JSON response."""
        try:
            # Grammar ensures valid JSON, but we still parse it
            data = json.loads(content)
            if not isinstance(data, list):
                data = data.get("rankings", [])

            results = []
            for item in data:
                # Ensure types are correct
                item["index"] = int(item["index"])
                item["score"] = float(item["score"])

                # Apply min_score filter (double-check)
                if item["score"] >= min_score:
                    if include_reasoning and "reason" not in item:
                        item["reason"] = ""
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
    print("=" * 60)
    print("GRAMMAR-BASED LLM RERANKING")
    print("=" * 60)
    print(f"Query: {query}\n")

    reranker = LLMReranker()

    # Test with reasoning DISABLED
    print("--- Testing with include_reasoning=False ---")
    results = reranker.rerank(
        query=query,
        documents=documents,
        top_k=5,
        min_score=7.0,
        include_reasoning=False,
    )

    if not results:
        print("No documents met the minimum score threshold.")
    else:
        for i, r in enumerate(results, 1):
            print(f"{i}. [score:{r['score']}/10] Index: {r['index']}")
            print()
