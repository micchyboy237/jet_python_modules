"""LLM-based reranker for complex relevance criteria with explainability."""

import json
import os
from typing import Optional, TypedDict

from openai import OpenAI


class RankingResult(TypedDict):
    index: int
    score: float
    document: str
    reason: str


class LLMReranker:
    """
    Use GPT-4 or Claude to rerank documents with explainable relevance.
    Best for complex relevance criteria and high-value result sets.
    """

    def __init__(
        self,
        model: str = os.getenv("LLAMA_CPP_LLM_MODEL", "not-needed"),
        base_url: Optional[str] = os.getenv("LLAMA_CPP_LLM_URL"),
        api_key: Optional[str] = "not-needed",
    ):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=120.0,
            max_retries=0,
        )
        self.model = model

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
        criteria: Optional[str] = None,
        return_explanation: bool = True,
    ) -> list[RankingResult]:
        """
        Rerank documents using LLM with detailed relevance scoring.
        Args:
            query: Search query
            documents: List of documents to rerank
            top_k: Number of results to return
            criteria: Additional relevance criteria for the LLM
            return_explanation: If True, include explanation for each ranking
        Returns:
            Ranked documents with scores and explanations
        """
        prompt = self._build_prompt(query, documents, criteria, return_explanation)
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a relevance ranking expert."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False,
                },
            },
            stream=True,
        )
        content = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    content += delta.content
                    print(delta.content, end="", flush=True)
        print()  # newline after stream
        try:
            result = json.loads(content)
            if isinstance(result, list):
                return result[:top_k]
            return result["rankings"][:top_k]
        except (json.JSONDecodeError, KeyError):
            return self._parse_text_response(content, documents)

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
        prompt = f"""Rank the following documents by their relevance to the query.
Query: {query}{criteria_text}
Documents:
{docs_text}
Return a JSON object with a "rankings" array. Each ranking should have:
- "index": original document index
- "score": relevance score from 0 to 10
- "document": the document text{', "reason": brief explanation' if return_explanation else ""}
Only return valid JSON, no other text."""
        return prompt

    def _parse_text_response(
        self, text: str, documents: list[str]
    ) -> list[RankingResult]:
        """Fallback parser for non-JSON responses."""
        return [
            {
                "index": 0,
                "score": 0,
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
