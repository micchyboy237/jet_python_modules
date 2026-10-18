"""
Demo: Chunking with Metadata and Vector Search
Shows how to generate ChunkResult objects for RAG indexing and find relevant chunks using embedding similarity.
"""

import shutil
from pathlib import Path
from typing import TypedDict

from jet.adapters.llama_cpp.chunking_utils import ChunkResult, chunk_texts_with_data
from jet.adapters.llama_cpp.config import (
    EMBED_DOC_PREFIX,
    EMBED_MODEL,
    EMBED_QUERY_PREFIX,
)
from jet.adapters.llama_cpp.vector_utils import vector_search
from rich.console import Console

console = Console()
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DOC_PATH = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/scrapers/playwright/generated/run_playwright_snapshot/snapshot.md"

import json


class ChunkMetadata(TypedDict):
    """Typed structure for chunk metadata."""

    chunk_index: int
    start_idx: int
    end_idx: int


class RankedChunkResult(TypedDict):
    """Typed structure for a ranked chunk result with metadata."""

    rank: int
    score: float
    text: str
    metadata: ChunkMetadata


def find_snapshot(
    query: str,
    text: str | list[str],
    top_k: int = 5,
    min_score: float = 0.2,
    model: str = EMBED_MODEL,
) -> tuple[list[RankedChunkResult], list[ChunkResult]]:
    """Find relevant chunks using vector similarity search.

    Args:
        query: Search query string.
        text: Source text or list of texts to chunk and search.
        top_k: Number of top results to return.
        min_score: Minimum cosine similarity threshold (0-1).
        model: Embedding model to use.

    Returns:
        Tuple of (ranked_results, all_chunks) where ranked_results contains:
        - rank: Result rank (1-based)
        - score: Cosine similarity score (0-1)
        - text: Chunk content
        - metadata: Chunk metadata (chunk_index, start_idx, end_idx)
    """
    console.print(
        "[cyan]Starting metadata-rich chunking demo with vector search...[/cyan]"
    )

    # Step 1: Chunk the text
    chunks = chunk_texts_with_data(
        texts=text,
        chunk_size=200,
        chunk_overlap=0,
        strict_sentences=True,
        show_progress=True,
        model=model,
    )

    if not chunks:
        console.print("[red]No chunks generated[/red]")
        return [], []

    documents = [chunk["content"] for chunk in chunks]
    metadata: list[ChunkMetadata] = [
        {
            "chunk_index": chunk["chunk_index"],
            "start_idx": chunk["start_idx"],
            "end_idx": chunk["end_idx"],
        }
        for chunk in chunks
    ]

    # Step 2: Use vector_search to embed and rank documents
    console.print(f"[dim]Running vector search on {len(documents)} documents...[/dim]")
    search_results = vector_search(
        query=query,
        documents=documents,
        top_n=top_k,
        query_prefix=EMBED_QUERY_PREFIX if EMBED_QUERY_PREFIX else None,
        document_prefix=EMBED_DOC_PREFIX if EMBED_DOC_PREFIX else None,
        embed_model=model,
    )

    # Step 3: Filter by min_score and enrich with metadata
    ranked_results: list[RankedChunkResult] = []
    for result in search_results:
        if result["score"] >= min_score:
            idx = result["index"]
            ranked_results.append(
                {
                    "rank": result["rank"],
                    "score": result["score"],
                    "text": result["text"],
                    "metadata": metadata[idx],
                }
            )

    console.print(
        f"[green]Found {len(ranked_results)} relevant chunks (min_score={min_score})[/green]"
    )

    return ranked_results, chunks


if __name__ == "__main__":
    query = "Search input"
    with open(DOC_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    ranked_results, chunks = find_snapshot(query, text)

    chunks_path = OUTPUT_DIR / "chunks.json"
    ranked_results_path = OUTPUT_DIR / "ranked_results.json"

    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    with open(ranked_results_path, "w", encoding="utf-8") as f:
        json.dump(ranked_results, f, indent=2, ensure_ascii=False)

    console.print(
        f"[bold green]✓ Generated {len(chunks)} chunks[/bold green] ([link=file://{chunks_path}]{chunks_path.name}[/link])"
    )
    console.print(
        f"[bold green]✓ Generated {len(ranked_results)} ranked results[/bold green] ([link=file://{ranked_results_path}]{ranked_results_path.name}[/link])"
    )

    # Display top results
    if ranked_results:
        console.print("\n[bold cyan]Top Results:[/bold cyan]")
        for result in ranked_results:
            score_display = f"score:{result['score']:.4f}"
            console.print(
                f"{result['rank']}. ({score_display}) Chunk {result['metadata']['chunk_index']} "
                f"(idx: {result['metadata']['start_idx']}-{result['metadata']['end_idx']})"
            )
            preview = result["text"][:100].replace("\n", " ")
            console.print(f"   Preview: {preview}...")
