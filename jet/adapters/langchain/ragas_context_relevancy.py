from typing import List, Dict, Any
from jet.adapters.langchain.chat_ollama import ChatOllama


class RagasContextRelevancy:
    """
    Custom implementation of Context Relevancy evaluation.
    Measures how relevant the retrieved contexts are to the query,
    using the LLM to score each context.
    """

    def __init__(self, model: str = "llama3.1", request_timeout: float = 300.0, context_window: int = 4096):
        self.model = ChatOllama(model=model, request_timeout=request_timeout)
        self.context_window = context_window

    def _score_context(self, query: str, context: str) -> float:
        """Ask the LLM to rate context relevance to the query (0–1)."""
        prompt = f"""
You are an evaluator. Rate how relevant the following context is to the given query.
Query: {query}
Context: {context}
Give a single float score between 0.0 (not relevant) and 1.0 (highly relevant).
Answer only with the number.
"""
        result = self.model.invoke(prompt)
        try:
            return float(result.content.strip())
        except Exception:
            return 0.0

    def run_batch(self, data: List[Dict[str, Any]]):
        """
        Expects data in the format:
        [
            {"query": "...", "response": "...", "context": ["...", "..."]},
            ...
        ]
        """
        results = []
        for record in data:
            query = record["query"]
            contexts = record.get("context", [])
            scores = [self._score_context(query, ctx) for ctx in contexts]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            results.append({
                "query": query,
                "response": record.get("response", ""),
                "context": contexts,
                "context_relevancy": avg_score
            })
        import pandas as pd
        return pd.DataFrame(results)
