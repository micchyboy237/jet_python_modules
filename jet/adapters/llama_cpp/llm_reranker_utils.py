"""LLM-based reranker using GBNF Grammar with Batched Token Processing."""

from __future__ import annotations

import json
from typing import List, Optional, Tuple, TypedDict, overload

from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.llama_cpp.factory import get_async_llm_client, get_llm_client
from jet.adapters.llama_cpp.llm_utils import achat, chat
from jet.adapters.llama_cpp.token_utils import count_tokens
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
    Splits documents into batches based on token limits.

    Design Pattern: Greedy Bin-Packing

    Algorithm:
    1. Start with an empty batch and 0 accumulated tokens.
    2. Iterate through documents.
    3. Add doc to current batch.
    4. If total tokens > max_tokens:
       - Save current batch as a completed group.
       - Start a new empty batch with the current doc.
    5. Return list of batches.
    """

    def __init__(self, model: str = LLM_MODEL):
        self.model = model
        logger.debug(f"BatchedTokenProcessor initialized with model={model}")

    def create_batches(
        self,
        documents: List[str],
        max_tokens: int = 500,
        query: Optional[str] = None,
    ) -> List[Tuple[List[str], List[int]]]:
        """
        Create batches of documents that fit within the token limit.

        Args:
            documents: List of document strings
            max_tokens: Maximum tokens per batch (including query overhead estimate)
            query: Optional query to account for in token budget

        Returns:
            List of tuples: (batch_documents, original_indices)
        """
        if not documents:
            return []

        # Estimate query tokens once
        query_tokens = count_tokens(query) if query else 0
        # Reserve some space for prompt structure/system message (approx 100 tokens)
        overhead = 100
        available_budget = max_tokens - query_tokens - overhead

        if available_budget <= 0:
            raise ValueError(
                "max_tokens is too low to accommodate query and prompt overhead."
            )

        batches = []
        current_batch_docs = []
        current_batch_indices = []
        current_batch_tokens = 0

        logger.info(f"Creating batches with budget {available_budget} tokens per batch")

        for i, doc in enumerate(documents):
            doc_tokens = count_tokens(doc, model=self.model)

            # If a single doc exceeds the entire budget, we have to include it alone
            # or raise an error. Here we include it alone but warn.
            if doc_tokens > available_budget:
                logger.warning(
                    f"Document[{i}] ({doc_tokens} tokens) exceeds batch budget "
                    f"({available_budget}). It will be processed in its own batch."
                )
                # If we have items in current batch, flush them first
                if current_batch_docs:
                    batches.append((current_batch_docs, current_batch_indices))
                    current_batch_docs = []
                    current_batch_indices = []
                    current_batch_tokens = 0

                batches.append(([doc], [i]))
                continue

            # Check if adding this doc exceeds budget
            if current_batch_tokens + doc_tokens > available_budget:
                # Flush current batch
                if current_batch_docs:
                    batches.append((current_batch_docs, current_batch_indices))

                # Start new batch
                current_batch_docs = [doc]
                current_batch_indices = [i]
                current_batch_tokens = doc_tokens
            else:
                # Add to current batch
                current_batch_docs.append(doc)
                current_batch_indices.append(i)
                current_batch_tokens += doc_tokens

        # Don't forget the last batch
        if current_batch_docs:
            batches.append((current_batch_docs, current_batch_indices))

        logger.info(f"Created {len(batches)} batches from {len(documents)} documents")
        return batches


class LLMReranker:
    """
    Use LLM to rerank documents using GBNF Grammar.

    Enhanced with batched processing to handle large document sets.
    """

    def __init__(
        self,
        model: str = LLM_MODEL,
        base_url: Optional[str] = None,
        api_key: Optional[str] = "not-needed",
        max_tokens: int = 500,
    ):
        self.model = model
        self._base_url = base_url
        self._api_key = api_key
        self.max_tokens = max_tokens
        self._processor = BatchedTokenProcessor(model=model)
        logger.debug(
            f"LLMReranker initialized with model={model}, max_tokens={max_tokens}"
        )

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
        max_tokens: Optional[int] = None,
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
        max_tokens: Optional[int] = None,
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
        max_tokens: Optional[int] = None,
    ) -> list[RankingResultDict | CompactRankingResultDict]:
        """Synchronous rerank with automatic batching."""

        limit = max_tokens if max_tokens is not None else self.max_tokens

        # 1. Create batches
        batches = self._processor.create_batches(
            documents=documents,
            max_tokens=limit,
            query=query,
        )

        all_results = []

        # 2. Process each batch
        for batch_idx, (batch_docs, original_indices) in enumerate(batches):
            logger.info(
                f"Processing batch {batch_idx + 1}/{len(batches)} ({len(batch_docs)} docs)"
            )

            prompt = self._build_grammar_prompt(
                query,
                batch_docs,
                criteria,
                max_doc_length,
                top_k,
                min_score,
                include_reasoning,
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

            # 3. Map local indices back to global original indices
            for res in batch_results:
                local_idx = res["index"]
                if local_idx < len(original_indices):
                    res_copy = dict(res)
                    res_copy["index"] = original_indices[local_idx]
                    all_results.append(res_copy)

        # 4. Sort all accumulated results by score descending
        all_results.sort(key=lambda x: x["score"], reverse=True)

        # 5. Return top_k overall
        return all_results[:top_k]

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
        max_tokens: Optional[int] = None,
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
        max_tokens: Optional[int] = None,
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
        max_tokens: Optional[int] = None,
    ) -> list[RankingResultDict | CompactRankingResultDict]:
        """Async rerank with automatic batching."""

        limit = max_tokens if max_tokens is not None else self.max_tokens

        # 1. Create batches
        batches = self._processor.create_batches(
            documents=documents,
            max_tokens=limit,
            query=query,
        )

        all_results = []

        # 2. Process each batch concurrently or sequentially
        # For simplicity and to avoid rate limits, we process sequentially here
        # but you could use asyncio.gather for concurrent processing
        for batch_idx, (batch_docs, original_indices) in enumerate(batches):
            logger.info(f"Async processing batch {batch_idx + 1}/{len(batches)}")

            prompt = self._build_grammar_prompt(
                query,
                batch_docs,
                criteria,
                max_doc_length,
                top_k,
                min_score,
                include_reasoning,
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

            # 3. Map local indices back to global original indices
            for res in batch_results:
                local_idx = res["index"]
                if local_idx < len(original_indices):
                    res_copy = dict(res)
                    res_copy["index"] = original_indices[local_idx]
                    all_results.append(res_copy)

        # 4. Sort all accumulated results by score descending
        all_results.sort(key=lambda x: x["score"], reverse=True)

        # 5. Return top_k overall
        return all_results[:top_k]

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
    print("GRAMMAR-BASED LLM RERANKING EVALUATION")
    print("=" * 60)
    print(f"Query: {query}\n")

    reranker = LLMReranker()

    print("--- TEST 1: include_reasoning=False ---")
    results_compact = reranker.rerank(
        query=query,
        documents=documents,
        top_k=5,
        min_score=7.0,
        include_reasoning=False,
    )
    if not results_compact:
        print("No documents met the minimum score threshold.")
    else:
        for i, r in enumerate(results_compact, 1):
            print(f"{i}. [score:{r['score']}/10] Index: {r['index']}")

    print()
    print("--- TEST 2: include_reasoning=True ---")
    results_full = reranker.rerank(
        query=query,
        documents=documents,
        top_k=5,
        min_score=7.0,
        criteria="Consider ecosystem maturity and library support",
        include_reasoning=True,
    )
    if not results_full:
        print("No documents met the minimum score threshold.")
    else:
        for i, r in enumerate(results_full, 1):
            print(f"{i}. [score:{r['score']}/10] Index: {r['index']}")
            if "reason" in r and r["reason"]:
                print(f"   Reason: {r['reason']}")

    print()
    print("=" * 60)
    print("Evaluation Complete")
    print("=" * 60)
