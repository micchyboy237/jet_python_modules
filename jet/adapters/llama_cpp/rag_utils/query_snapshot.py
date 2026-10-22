# jet_python_modules/jet/adapters/llama_cpp/rag_utils/query_snapshot.py
"""
Demo: Chunking with Metadata and LLM Reranking
Shows how to generate ChunkResult objects for RAG indexing and find relevant chunks using LLM-based reranking.
"""

import shutil
from pathlib import Path
from typing import Any, TypedDict

from jet.adapters.llama_cpp.chunking_utils import ChunkResult, chunk_texts_with_data
from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.llama_cpp.llm_reranker_utils import rerank
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


def format_chunk_for_rag(
    rank: int, score: float, text: str, metadata: ChunkMetadata
) -> str:
    """Format a single chunk for LLM RAG context injection in Markdown.
    Args:
        rank: Result rank (1-based).
        score: Relevance score (0-10).
        text: Chunk content.
        metadata: Chunk metadata.
    Returns:
        Formatted markdown string suitable for LLM context.
    """
    return (
        f"### Document {rank} (Relevance: {score:.1f}/10)\n"
        f"- **Chunk Index**: {metadata['chunk_index']}\n"
        f"- **Position**: Characters {metadata['start_idx']}–{metadata['end_idx']}\n"
        f"\n"
        f"{text}\n"
    )


def find_snapshot(
    query: str,
    text: str | list[str],
    top_k: int = 5,
    min_score: float = 7.0,
    model: str = LLM_MODEL,
) -> tuple[str, list[RankedChunkResult], list[ChunkResult]]:
    """Find relevant chunks using LLM-based reranking.
    Args:
        query: Search query string.
        text: Source text or list of texts to chunk and search.
        top_k: Number of top results to return.
        min_score: Minimum relevance score threshold (0-10).
        model: LLM model to use for reranking.
    Returns:
        Tuple of (rag_context, ranked_results, all_chunks) where:
        - rag_context: Joined formatted chunks optimized for LLM RAG (Markdown)
        - ranked_results: List of ranked chunk results with metadata
        - all_chunks: All generated chunks from the source text
    """
    console.print(
        "[cyan]Starting metadata-rich chunking demo with LLM reranking...[/cyan]"
    )
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
        return "", [], []

    documents = [chunk["content"] for chunk in chunks]
    metadata: list[ChunkMetadata] = [
        {
            "chunk_index": chunk["chunk_index"],
            "start_idx": chunk["start_idx"],
            "end_idx": chunk["end_idx"],
        }
        for chunk in chunks
    ]

    # Prepare metadata for reranking (optional, but useful for context)
    # We can pass the metadata list to the reranker so it's included in results
    meta_dicts: list[dict[str, Any]] = [
        {
            "chunk_index": m["chunk_index"],
            "start_idx": m["start_idx"],
            "end_idx": m["end_idx"],
        }
        for m in metadata
    ]

    console.print(f"[dim]Running LLM reranking on {len(documents)} documents...[/dim]")

    # Use llm_reranker_utils.rerank
    # Note: rerank returns results sorted by score descending, with 'index' referring to original position
    search_results = rerank(
        query=query,
        documents=documents,
        metadata=meta_dicts,
        top_k=top_k,
        min_score=min_score,
        include_reasoning=False,  # Use CompactRankingResult for efficiency
        model=model,
    )

    ranked_results: list[RankedChunkResult] = []
    rag_fragments: list[str] = []

    # search_results is already sorted by score descending and filtered by min_score/top_k
    for rank_position, result in enumerate(search_results, start=1):
        idx = result["index"]
        # The reranker might return metadata if provided
        chunk_metadata = result.get("metadata", metadata[idx])

        # Ensure we have the correct typed metadata
        if isinstance(chunk_metadata, dict):
            typed_meta: ChunkMetadata = {
                "chunk_index": chunk_metadata.get(
                    "chunk_index", metadata[idx]["chunk_index"]
                ),
                "start_idx": chunk_metadata.get(
                    "start_idx", metadata[idx]["start_idx"]
                ),
                "end_idx": chunk_metadata.get("end_idx", metadata[idx]["end_idx"]),
            }
        else:
            typed_meta = metadata[idx]

        ranked_results.append(
            {
                "rank": rank_position,
                "score": result["score"],
                "text": result["text"],
                "metadata": typed_meta,
            }
        )
        rag_fragments.append(
            format_chunk_for_rag(
                rank=rank_position,
                score=result["score"],
                text=result["text"],
                metadata=typed_meta,
            )
        )

    rag_context = ""
    if rag_fragments:
        rag_context = (
            f'# Retrieved Context for Query: "{query}"\n\n'
            f"> Found {len(ranked_results)} relevant chunks with minimum relevance score of {min_score}/10\n\n"
            "---\n\n"
        ) + "\n---\n\n".join(rag_fragments)

    console.print(
        f"[green]Found {len(ranked_results)} relevant chunks (min_score={min_score})[/green]"
    )
    return rag_context, ranked_results, chunks


if __name__ == "__main__":
    query = "Search input"
    with open(DOC_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    rag_context, ranked_results, chunks = find_snapshot(query, text)

    chunks_path = OUTPUT_DIR / "chunks.json"
    ranked_results_path = OUTPUT_DIR / "ranked_results.json"
    rag_context_path = OUTPUT_DIR / "rag_context.md"

    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    with open(ranked_results_path, "w", encoding="utf-8") as f:
        json.dump(ranked_results, f, indent=2, ensure_ascii=False)

    with open(rag_context_path, "w", encoding="utf-8") as f:
        f.write(rag_context)

    if ranked_results:
        console.print("\n[bold cyan]Top Results:[/bold cyan]")
        for result in ranked_results:
            score_display = f"score:{result['score']:.1f}/10"
            console.print(
                f"{result['rank']}. ({score_display}) Chunk {result['metadata']['chunk_index']} "
                f"(idx: {result['metadata']['start_idx']}-{result['metadata']['end_idx']})"
            )
            preview = result["text"][:100].replace("\n", " ")
            console.print(f"   Preview: {preview}...")

    if rag_context:
        console.print("\n[bold cyan]RAG Context Preview (Markdown):[/bold cyan]")
        preview_lines = rag_context.split("\n")[:20]
        for line in preview_lines:
            console.print(f"  [dim]{line}[/dim]")
        if len(rag_context.split("\n")) > 20:
            console.print("  [dim]...[/dim]")

    console.print(
        f"[bold green]✓ Generated {len(chunks)} chunks[/bold green] ([link=file://{chunks_path}]{chunks_path.name}[/link])"
    )
    console.print(
        f"[bold green]✓ Generated {len(ranked_results)} ranked results[/bold green] ([link=file://{ranked_results_path}]{ranked_results_path.name}[/link])"
    )
    console.print(
        f"[bold green]✓ Generated RAG context ({len(rag_context)} chars)[/bold green] ([link=file://{rag_context_path}]{rag_context_path.name}[/link])"
    )
