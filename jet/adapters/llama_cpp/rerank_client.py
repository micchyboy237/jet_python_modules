"""
BGE Reranker Client for llama.cpp server
Uses environment variables for configuration
"""

import os
from dataclasses import dataclass
from typing import List, Optional, TypedDict

import requests

# Configuration from environment variables with defaults
RERANK_BASE_URL = os.getenv("LLAMA_CPP_RERANK_URL", "http://localhost:8082/v1")
RERANK_URL = RERANK_BASE_URL + "/rerank"
MODEL = os.getenv("LLAMA_CPP_RERANK_MODEL")


# Type definitions
class RerankResult(TypedDict):
    """Single rerank result from llama.cpp"""

    index: int
    relevance_score: float


class RerankResponse(TypedDict):
    """Full rerank API response"""

    results: List[RerankResult]


@dataclass
class RankedDocument:
    """Convenient wrapper for ranked documents"""

    index: int
    score: float
    text: str

    def __str__(self) -> str:
        return f"[Score: {self.score:.4f}] {self.text[:100]}..."


class RerankerClient:
    """Client for BGE reranker running on llama.cpp server"""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: int = 30,
    ):
        """
        Initialize reranker client

        Args:
            base_url: Base URL of llama.cpp server (default from env)
            model_name: Model name to use (default from env)
            timeout: Request timeout in seconds
        """
        # Use provided values or fall back to module-level defaults
        self.base_url = base_url or RERANK_BASE_URL
        self.model_name = model_name or MODEL
        self.timeout = timeout
        self.rerank_url = (
            f"{self.base_url}/rerank"
            if not self.base_url.endswith("/rerank")
            else self.base_url
        )

        # Validate configuration
        if not self.base_url:
            raise ValueError("Base URL must be provided")
        if not self.model_name:
            raise ValueError(
                "Model name must be provided. Set LLAMA_CPP_RERANK_MODEL environment variable "
                "or pass model_name parameter."
            )

    def rerank_documents(
        self, query: str, documents: List[str], top_n: Optional[int] = None
    ) -> List[RankedDocument]:
        """
        Rerank documents based on relevance to query

        Args:
            query: The search query
            documents: List of document texts to rerank
            top_n: Number of top results to return (None = all)

        Returns:
            List of RankedDocument sorted by relevance score (descending)

        Raises:
            requests.exceptions.RequestException: If API call fails
            ValueError: If documents list is empty or model not configured
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")
        if not query.strip():
            raise ValueError("Query cannot be empty")

        # Prepare payload
        payload = {"model": self.model_name, "query": query, "documents": documents}

        if top_n is not None:
            payload["top_n"] = top_n

        # Make API request
        try:
            response = requests.post(
                self.rerank_url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to reranker at {self.rerank_url}. "
                "Is the llama.cpp server running?"
            )
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request to reranker timed out after {self.timeout}s")

        # Parse response
        data: RerankResponse = response.json()
        results = data.get("results", [])

        if not results:
            return []

        # Map results back to original documents using index
        ranked_docs = []
        for result in results:
            idx = result["index"]
            if 0 <= idx < len(documents):
                ranked_docs.append(
                    RankedDocument(
                        index=idx, score=result["relevance_score"], text=documents[idx]
                    )
                )

        # Sort by score descending (should already be sorted, but ensure)
        ranked_docs.sort(key=lambda x: x.score, reverse=True)

        return ranked_docs

    def get_top_documents(
        self, query: str, documents: List[str], top_k: int = 5
    ) -> List[RankedDocument]:
        """
        Convenience method to get top-k most relevant documents

        Args:
            query: The search query
            documents: List of document texts
            top_k: Number of top documents to return

        Returns:
            Top-k ranked documents
        """
        return self.rerank_documents(query, documents, top_n=top_k)

    def format_context(
        self,
        ranked_docs: List[RankedDocument],
        max_docs: int = 3,
        separator: str = "\n\n",
    ) -> str:
        """
        Format ranked documents into context string for LLM

        Args:
            ranked_docs: List of ranked documents
            max_docs: Maximum number of documents to include
            separator: Separator between documents

        Returns:
            Formatted context string
        """
        selected = ranked_docs[:max_docs]
        return separator.join(doc.text for doc in selected)


def demo():
    """Demonstration of reranker usage"""

    print("=" * 70)
    print("BGE Reranker Demo")
    print("=" * 70)
    print(f"Base URL: {RERANK_BASE_URL}")
    print(f"Rerank URL: {RERANK_URL}")
    print(f"Model: {MODEL or 'Not set (will use default)'}")
    print("=" * 70)

    # Initialize client
    try:
        client = RerankerClient()
        print(f"\n✓ Connected to: {client.rerank_url}")
        print(f"✓ Using model: {client.model_name}\n")
    except Exception as e:
        print(f"✗ Failed to initialize client: {e}")
        return

    # Sample query and documents
    query = "How does photosynthesis work?"

    retrieved_docs = [
        "Photosynthesis is the process by which plants convert sunlight into energy.",
        "The mitochondria is the powerhouse of the cell.",
        "Chlorophyll absorbs light energy to convert CO2 and water into glucose.",
        "Python is a popular programming language for data science.",
        "Plants release oxygen as a byproduct of photosynthesis.",
        "The stock market experienced significant volatility last quarter.",
        "Light-dependent reactions occur in the thylakoid membranes.",
        "Basketball was invented by James Naismith in 1891.",
        "The Calvin cycle uses ATP and NADPH to produce sugars.",
        "Machine learning algorithms can improve with more data.",
    ]

    print(f"Query: '{query}'\n")
    print(f"Retrieved {len(retrieved_docs)} documents for reranking...\n")

    # Rerank documents
    try:
        ranked_docs = client.get_top_documents(
            query=query, documents=retrieved_docs, top_k=5
        )

        print("-" * 70)
        print("TOP 5 RERANKED DOCUMENTS:")
        print("-" * 70)

        for i, doc in enumerate(ranked_docs, 1):
            print(f"\n{i}. Score: {doc.score:.4f} (Original index: {doc.index})")
            print(f"   Text: {doc.text}")

        # Format context for LLM
        print("\n" + "-" * 70)
        print("FORMATTED CONTEXT FOR LLM (Top 3):")
        print("-" * 70)
        context = client.format_context(ranked_docs, max_docs=3)
        print(context)

        print("\n" + "=" * 70)
        print("Demo completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Error during reranking: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    demo()
