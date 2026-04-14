import json
from typing import Any, Dict, List

from jet.libs.crawl4ai_lib.async_web_crawler_manager import AsyncWebCrawlerManager
from jet.libs.crawl4ai_lib.search_searxng import semantic_search_results
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SlidingWindowChunking:
    """Exactly the class from chunking.md - creates overlapping chunks for better context."""

    def __init__(self, window_size=100, step=50):
        self.window_size = window_size
        self.step = step

    def chunk(self, text: str) -> List[str]:
        if not text:
            return []
        words = text.split()
        chunks = []
        for i in range(0, len(words) - self.window_size + 1, self.step):
            chunks.append(" ".join(words[i : i + self.window_size]))
        return chunks


class CosineSimilarityExtractor:
    """Exactly the class from chunking.md - scores chunks against your query."""

    def __init__(self, query: str):
        self.query = query
        self.vectorizer = TfidfVectorizer()

    def find_relevant_chunks(self, chunks: List[str]) -> List[tuple[str, float]]:
        if not chunks:
            return []
        vectors = self.vectorizer.fit_transform([self.query] + chunks)
        similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
        return [(chunks[i], float(similarities[i])) for i in range(len(chunks))]


class CombinedRAGChunker:
    """Combined approach from chunking.md: sliding window + cosine similarity.
    This is the new recommended chunking logic for RAG."""

    def __init__(self, window_size: int = 100, step: int = 50):
        self.window_size = window_size
        self.step = step

    def get_relevant_chunks(
        self, text: str, user_query: str, top_k: int = 10
    ) -> List[str]:
        """1. Cut into overlapping windows (SlidingWindowChunking)
        2. Score with cosine similarity (CosineSimilarityExtractor)
        3. Return the best matching chunks"""
        if not text or not user_query:
            return []

        # Step 1: Sliding window chunking
        chunker = SlidingWindowChunking(self.window_size, self.step)
        chunks = chunker.chunk(text)

        # Step 2: Cosine similarity ranking
        extractor = CosineSimilarityExtractor(user_query)
        scored_chunks = extractor.find_relevant_chunks(chunks)

        # Sort by score (highest first) and keep only top_k
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, score in scored_chunks[:top_k]]


class RAGCrawler:
    """Easy RAG helper that reuses the existing crawler manager."""

    def __init__(self):
        self.crawler_manager = AsyncWebCrawlerManager(
            headless=True,
            verbose=False,
            cache_mode="BYPASS",  # fresh data for RAG
        )

    async def run_rag_query(self, user_query: str):
        """Full RAG flow: crawl → BM25 markdown → chunk → return chunks."""
        print(f"🚀 RAG query: '{user_query}'")
        urls = semantic_search_results(user_query)
        # The manager already knows how to use BM25 when user_query is passed
        # In a real app you would collect results here; this is the pattern
        print("   → Crawling with BM25 filter → fit_markdown → combined chunking")
        # For demo we return placeholder - replace with real streaming logic
        demo_fit_markdown = "This is the cleaned fit_markdown from the page.\n\nIt contains the important parts about Crawl4AI and RAG."
        chunker = CombinedRAGChunker(window_size=80, step=40)
        demo_chunks = chunker.get_relevant_chunks(demo_fit_markdown, user_query)
        return {"fit_markdown": demo_fit_markdown, "chunks": demo_chunks}


def run_generate_browser_query_context_json_schema(
    user_query: str,
    retrieved_chunks: List[str] | None = None,
    fit_markdown: str | None = None,
) -> Dict[str, Any]:
    """
    Creates the JSON schema for the "context" part of a browser query in RAG.
    This tells the AI: "Here is the background info from the web".

    FULL USAGE EXAMPLE (copy and run):
    ```python
    from rag_crawler import run_generate_browser_query_context_json_schema
    import json

    schema = run_generate_browser_query_context_json_schema(
        user_query="How does Crawl4AI help with RAG?",
        retrieved_chunks=["Chunk 1: markdown is clean...", "Chunk 2: BM25 keeps relevant parts"],
        fit_markdown="## RAG Basics\nClean text from web pages..."
    )
    print(json.dumps(schema, indent=2))
    ```
    """
    if retrieved_chunks is None:
        retrieved_chunks = []
    if fit_markdown is None:
        fit_markdown = ""

    schema = {
        "type": "object",
        "properties": {
            "user_query": {
                "type": "string",
                "description": f"The original question the user asked: {user_query}",
            },
            "retrieved_chunks": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Best chunks picked after chunking the fit_markdown (using cosine similarity or BM25)",
            },
            "fit_markdown_summary": {
                "type": "string",
                "description": "Clean markdown produced by markdown-generation.md (BM25 or Pruning filter)",
            },
        },
        "required": ["user_query", "retrieved_chunks"],
    }
    return schema


def run_generate_browser_query_json_schema(
    required_fields: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Creates the JSON schema for the final answer after the browser query.
    This forces the AI to return data in a clean, predictable format.

    FULL USAGE EXAMPLE (copy and run):
    ```python
    from rag_crawler import run_generate_browser_query_json_schema
    import json

    schema = run_generate_browser_query_json_schema(
        required_fields=["title", "summary", "key_points"]
    )
    print(json.dumps(schema, indent=2))
    ```
    """
    if required_fields is None:
        required_fields = ["title", "summary"]

    schema = {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Title of the web page (from metadata)",
            },
            "summary": {
                "type": "string",
                "description": "Short summary created from the fit_markdown",
            },
            "key_points": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Main ideas pulled from the chunks",
            },
        },
        "required": required_fields,
    }
    return schema


def run_generate_field_descriptions(fields: List[str] | None = None) -> Dict[str, str]:
    """
    Gives plain-English descriptions for every field.
    Super useful when building prompts or explaining the schema to another AI.

    FULL USAGE EXAMPLE (copy and run):
    ```python
    from rag_crawler import run_generate_field_descriptions
    import json

    descriptions = run_generate_field_descriptions(
        fields=["title", "summary", "key_points"]
    )
    print(json.dumps(descriptions, indent=2))
    ```
    """
    if fields is None:
        fields = ["title", "summary", "key_points"]

    all_desc = {
        "title": "The main title of the webpage - easy to spot at the top.",
        "summary": "A short, clear summary of the important content (comes from fit_markdown after chunking).",
        "key_points": "Bullet-list of the most useful ideas found in the chunks.",
        "chunks": "Small pieces of clean markdown ready for RAG retrieval.",
    }
    return {
        field: all_desc.get(field, f"Description for field '{field}'")
        for field in fields
    }


# Quick demo when you run the file directly
if __name__ == "__main__":
    print("=== RAG Schema Helpers - Full Usage Demo ===\n")
    ctx = run_generate_browser_query_context_json_schema(
        user_query="How does chunking help RAG?",
        retrieved_chunks=["Chunk 1: cut text into pieces", "Chunk 2: find best match"],
    )
    q_schema = run_generate_browser_query_json_schema(["title", "summary"])
    desc = run_generate_field_descriptions()

    print("Context Schema:")
    print(json.dumps(ctx, indent=2))
    print("\nBrowser Query Schema:")
    print(json.dumps(q_schema, indent=2))
    print("\nField Descriptions:")
    print(json.dumps(desc, indent=2))
    print("\n✅ All three functions ready for your RAG pipeline!")
