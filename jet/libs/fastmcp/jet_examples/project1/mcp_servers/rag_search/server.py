"""Simple RAG-style search server using embeddings."""
import os
from typing import List, Dict, Any
from fastmcp import FastMCP
from pydantic import BaseModel

mcp = FastMCP(
    name="RAGSearch",
    instructions="You are a document search assistant using vector embeddings."
)

class SearchInput(BaseModel):
    query: str
    top_k: int = 3

@mcp.tool(output_schema=List[Dict[str, Any]])
def semantic_search(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Perform semantic search over documents (mocked)."""
    # In real life → call your embeddings endpoint + vector DB
    mock_results = [
        {"id": "doc1", "text": f"Mock result 1 for '{query}'", "score": 0.95},
        {"id": "doc2", "text": f"Mock result 2 for '{query}'", "score": 0.88},
        {"id": "doc3", "text": f"Mock result 3 for '{query}'", "score": 0.75},
    ]
    return mock_results[:top_k]

@mcp.prompt(name="rag_answer")
def rag_answer_template(query: str, context: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Prompt template for answering with retrieved context."""
    context_str = "\n".join([f"- {r['text']} (score: {r['score']})" for r in context])
    return [
        {"role": "system", "content": "Answer using only the provided context."},
        {"role": "user", "content": f"Question: {query}\n\nContext:\n{context_str}"}
    ]

if __name__ == "__main__":
    mcp.run()
