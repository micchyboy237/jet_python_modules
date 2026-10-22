# jet_python_modules/jet/adapters/llama_cpp/rag_utils/query_snapshot.py
"""
Demo: Chunking with Metadata and LLM Reranking
Shows how to generate ChunkResult objects for RAG indexing and find relevant chunks using LLM-based reranking.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, TypedDict

from jet.adapters.llama_cpp.chunking_utils import ChunkResult, chunk_texts_with_data
from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.llama_cpp.llm_reranker_utils import rerank
from rich.console import Console

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem


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


def get_args() -> argparse.Namespace:
    """Parse command-line arguments with complete rerank parameters."""
    parser = argparse.ArgumentParser(
        description="Chunk documents and rerank using LLM-based relevance scoring.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -q "What is Python?" -d document.md
  %(prog)s -q "search query" -d doc.md -k 10 -m 8.0 --model qwen3.5-uncensored:2b
  %(prog)s -q "query" -d doc.md --include-reasoning --criteria "focus on technical details"
        """,
    )

    # Core inputs
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        default="Search and location inputs",
        help="Search query string for reranking (default: 'Search and location inputs').",
    )
    parser.add_argument(
        "-d",
        "--doc-path",
        type=str,
        default="/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/scrapers/playwright/generated/run_playwright_agent/steps/step_0/initial_snapshot.md",
        help="Path to the source document (markdown or text file).",
    )

    # Chunking parameters
    parser.add_argument(
        "-cs",
        "--chunk-size",
        type=int,
        default=200,
        help="Maximum tokens per chunk (default: 200). This also sets the max tokens for reranking.",
    )
    parser.add_argument(
        "-co",
        "--chunk-overlap",
        type=int,
        default=0,
        help="Number of overlapping tokens between chunks (default: 0).",
    )
    parser.add_argument(
        "--strict-sentences",
        action="store_true",
        default=True,
        help="Respect sentence boundaries during chunking (default: True).",
    )

    # Reranking parameters
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=None,
        help="Number of top results to return (default: None).",
    )
    parser.add_argument(
        "-m",
        "--min-score",
        type=float,
        default=7.0,
        help="Minimum relevance score threshold 0-10 (default: 7.0).",
    )
    parser.add_argument(
        "--criteria",
        type=str,
        default=None,
        help="Optional specific criteria for the LLM to focus on during ranking.",
    )
    parser.add_argument(
        "--include-reasoning",
        action="store_true",
        default=False,
        help="If set, LLM provides a reason for each ranking (default: False).",
    )
    parser.add_argument(
        "--batch-overlap",
        type=int,
        default=0,
        help="Number of documents to overlap between batches for continuity (default: 0).",
    )

    # Model and server configuration
    parser.add_argument(
        "--model",
        type=str,
        default=LLM_MODEL,
        help=f"LLM model to use for reranking (default: {LLM_MODEL}).",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL for the LLM server (default: auto-detect from config).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="not-needed",
        help="API key for the LLM server (default: 'not-needed').",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8000,
        help="Max context window tokens for the LLM. If None, auto-detected (default: 8000).",
    )

    # Output configuration
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help=f"Output directory for generated files (default: {OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        default=False,
        help="Disable progress bars during processing.",
    )

    return parser.parse_args()


def find_snapshot(
    query: str,
    text: str | list[str],
    top_k: int | None = None,
    min_score: float = 7.0,
    model: str = LLM_MODEL,
    criteria: str | None = None,
    include_reasoning: bool = False,
    base_url: str | None = None,
    api_key: str | None = "not-needed",
    max_tokens: int | None = 8000,
    batch_overlap: int = 0,
    chunk_size: int = 200,
    chunk_overlap: int = 0,
    strict_sentences: bool = True,
    show_progress: bool = True,
) -> tuple[str, list[RankedChunkResult], list[ChunkResult]]:
    """Find relevant chunks using LLM-based reranking.
    Args:
        query: Search query string.
        text: Source text or list of texts to chunk and search.
        top_k: Number of top results to return.
        min_score: Minimum relevance score threshold (0-10).
        model: LLM model to use for reranking.
        criteria: Optional specific criteria for the LLM to focus on.
        include_reasoning: If True, LLM provides a reason for each ranking.
        base_url: Base URL for the LLM server.
        api_key: API key for the LLM server.
        max_tokens: Max context window tokens for the LLM. If None, auto-detected.
        batch_overlap: Number of documents to overlap between batches.
        chunk_size: Maximum tokens per chunk. Also used as max_doc_tokens for reranking.
        chunk_overlap: Number of overlapping tokens between chunks.
        strict_sentences: If True, respect sentence boundaries.
        show_progress: Show progress bar during processing.
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
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strict_sentences=strict_sentences,
        show_progress=show_progress,
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
    # Note: max_doc_tokens is set to chunk_size to ensure no context is lost during truncation
    search_results = rerank(
        query=query,
        documents=documents,
        metadata=meta_dicts,
        top_k=top_k,
        min_score=min_score,
        criteria=criteria,
        include_reasoning=include_reasoning,
        model=model,
        base_url=base_url,
        api_key=api_key,
        max_tokens=max_tokens,
        batch_overlap=batch_overlap,
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
    args = get_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read document
    console.print(f"[dim]Reading document from: {args.doc_path}[/dim]")
    with open(args.doc_path, "r", encoding="utf-8") as f:
        text = f.read()

    rag_context, ranked_results, chunks = find_snapshot(
        query=args.query,
        text=text,
        top_k=args.top_k,
        min_score=args.min_score,
        model=args.model,
        criteria=args.criteria,
        include_reasoning=args.include_reasoning,
        base_url=args.base_url,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        batch_overlap=args.batch_overlap,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        strict_sentences=args.strict_sentences,
        show_progress=not args.no_progress,
    )

    chunks_path = output_dir / "chunks.json"
    ranked_results_path = output_dir / "ranked_results.json"
    rag_context_path = output_dir / "rag_context.md"

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
