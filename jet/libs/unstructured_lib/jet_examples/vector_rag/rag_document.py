"""Reusable typed document for the entire RAG pipeline - generic, no business logic."""
from dataclasses import dataclass
from typing import Dict, Any, List, TypedDict

class Chunk(TypedDict):
    """Minimal, reusable chunk structure with text + metadata (from unstructured elements)."""
    text: str
    metadata: Dict[str, Any]

@dataclass
class RAGDocument:
    """Optional dataclass wrapper for type safety in advanced usage (swapable with TypedDict)."""
    page_content: str
    metadata: Dict[str, Any]

RAGDocumentList = List[RAGDocument]
ChunkList = List[Chunk]
