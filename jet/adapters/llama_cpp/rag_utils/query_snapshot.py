"""
Demo: Chunking with Metadata and Vector-Filtered LLM Reranking
Shows how to generate ChunkResult objects, filter via Vector Search,
and then rerank using LLM-based relevance scoring.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import TypedDict

from jet.adapters.llama_cpp.chunking_utils import ChunkResult, chunk_texts_with_data
from jet.adapters.llama_cpp.config import (
    EMBED_DOC_PREFIX,
    EMBED_MODEL,
    EMBED_QUERY_PREFIX,
    LLM_MODEL,
)
from jet.adapters.llama_cpp.llm_reranker_utils import rerank as llm_rerank
from jet.adapters.llama_cpp.token_utils import count_tokens
from jet.adapters.llama_cpp.vector_utils import vector_search
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
    """Format a single chunk for LLM RAG context injection in Markdown."""
    return (
        f"### Document {rank} (Relevance: {score:.1f}/10)\n"
        f"- **Chunk Index**: {metadata['chunk_index']}\n"
        f"- **Position**: Characters {metadata['start_idx']}–{metadata['end_idx']}\n"
        "\n"
        f"{text}\n"
    )


def get_args() -> argparse.Namespace:
    """Parse command-line arguments with complete rerank parameters."""
    parser = argparse.ArgumentParser(
        description="Chunk documents, filter via Vector Search, and rerank using LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        default="Search and location inputs",
        help="Search query string.",
    )
    parser.add_argument(
        "-d",
        "--doc-path",
        type=str,
        default="/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/scrapers/playwright/generated/run_playwright_agent/steps/step_0/initial_snapshot.md",
        help="Path to the source document. Can also be a JSON file with an array of strings.",
    )
    parser.add_argument(
        "-cs", "--chunk-size", type=int, default=500, help="Max tokens per chunk."
    )
    parser.add_argument(
        "-co", "--chunk-overlap", type=int, default=100, help="Overlap tokens."
    )
    parser.add_argument(
        "--strict-sentences",
        action="store_true",
        default=True,
        help="Respect sentence boundaries.",
    )
    parser.add_argument(
        "-ck",
        "--candidate-k",
        type=int,
        default=10,
        help="Number of top candidates from Vector Search to pass to LLM Reranker (default: 10).",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=10,
        help="Final number of top results to return.",
    )
    parser.add_argument(
        "-m",
        "--min-score",
        type=float,
        default=3.0,
        help="Minimum relevance score threshold.",
    )
    parser.add_argument(
        "--min-results",
        type=int,
        default=4,
        help="Stop processing batches once this many results are found (early exit).",
    )
    parser.add_argument(
        "--min-high-results",
        type=int,
        default=2,
        help="Stop processing batches once this many high-quality (high-score) results are found (early exit).",
    )
    parser.add_argument(
        "--criteria", type=str, default=None, help="Optional criteria for LLM."
    )
    parser.add_argument(
        "--no-reasoning",
        action="store_true",
        default=False,
        help="Disable LLM reasoning.",
    )
    parser.add_argument(
        "--batch-overlap", type=int, default=0, help="Batch overlap for LLM reranker."
    )
    parser.add_argument(
        "--model", type=str, default=LLM_MODEL, help="LLM model for reranking."
    )
    parser.add_argument(
        "--base-url", type=str, default=None, help="Base URL for LLM server."
    )
    parser.add_argument(
        "--api-key", type=str, default="not-needed", help="API key for LLM server."
    )
    parser.add_argument(
        "--max-tokens", type=int, default=8000, help="Max context window tokens."
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Output directory.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        default=False,
        help="Disable progress bars.",
    )
    return parser.parse_args()


def find_snapshot(
    query: str,
    text: str | list[str],
    top_k: int | None = 10,
    min_score: float = 3.0,
    min_results: int | None = 4,
    min_high_results: int | None = 2,
    model: str = LLM_MODEL,
    embed_model: str = EMBED_MODEL,
    query_prefix: str | None = EMBED_QUERY_PREFIX,
    document_prefix: str | None = EMBED_DOC_PREFIX,
    criteria: str | None = None,
    include_reasoning: bool = True,
    base_url: str | None = None,
    api_key: str | None = "not-needed",
    max_tokens: int | None = 8000,
    batch_overlap: int = 0,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    embed_chunk_size: int = 300,
    embed_chunk_overlap: int = 100,
    strict_sentences: bool = True,
    show_progress: bool = True,
    candidate_k: int = 10,
) -> tuple[str, list[RankedChunkResult], list[ChunkResult]]:
    """Find relevant chunks using Vector Search filtering followed by LLM Reranking.
    Args:
        query: Search query string.
        text: Source text or list of texts.
        top_k: Final number of top results to return after LLM reranking.
        min_score: Minimum relevance score threshold for LLM reranker.
        min_results: If set, stop processing batches once this many results are found.
        min_high_results: If set, stop processing batches once this many high-quality (high-score) results are found.
        model: LLM model for reranking.
        criteria: Optional criteria for LLM.
        include_reasoning: If True, LLM provides reasoning.
        base_url: Base URL for LLM server.
        api_key: API key for LLM server.
        max_tokens: Max context window tokens.
        batch_overlap: Batch overlap for LLM reranker.
        chunk_size: Max tokens per chunk.
        chunk_overlap: Overlap tokens.
        strict_sentences: Respect sentence boundaries.
        show_progress: Show progress bars.
        candidate_k: Number of top candidates from Vector Search to pass to LLM Reranker.
    Returns:
        Tuple of (rag_context, ranked_results, all_chunks)
    """
    console.print(
        "[cyan]Starting vector-filtered chunking demo with LLM reranking...[/cyan]"
    )
    chunks = chunk_texts_with_data(
        texts=text,
        chunk_size=embed_chunk_size,
        chunk_overlap=embed_chunk_overlap,
        strict_sentences=strict_sentences,
        show_progress=show_progress,
        model=embed_model,
    )
    if not chunks:
        console.print("[red]No chunks generated[/red]")
        return "", [], []

    documents = [chunk["content"] for chunk in chunks]
    metadata_list: list[ChunkMetadata] = [
        {
            "chunk_index": chunk["chunk_index"],
            "start_idx": chunk["start_idx"],
            "end_idx": chunk["end_idx"],
        }
        for chunk in chunks
    ]

    console.print(
        f"[dim]Generated {len(documents)} chunks. Running Vector Search for top {candidate_k} candidates...[/dim]"
    )
    vector_results = vector_search(
        query=query,
        documents=documents,
        top_n=candidate_k,
        embed_model=embed_model,
        query_prefix=query_prefix,
        document_prefix=document_prefix,
    )
    if not vector_results:
        console.print("[red]Vector search returned no results.[/red]")
        return "", [], chunks

    console.print(
        f"[green]Vector Search retrieved {len(vector_results)} candidates.[/green]"
    )
    candidate_indices = [r["index"] for r in vector_results]
    candidate_docs = []
    candidate_meta_dicts = []
    for idx in candidate_indices:
        candidate_docs.append(documents[idx])
        candidate_meta_dicts.append(
            {
                "chunk_index": metadata_list[idx]["chunk_index"],
                "start_idx": metadata_list[idx]["start_idx"],
                "end_idx": metadata_list[idx]["end_idx"],
            }
        )

    console.print(
        f"[dim]Running LLM reranking on {len(candidate_docs)} candidates...[/dim]"
    )
    search_results = llm_rerank(
        query=query,
        documents=candidate_docs,
        metadata=candidate_meta_dicts,
        top_k=top_k,
        min_score=min_score,
        min_results=min_results,
        min_high_results=min_high_results,
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
    for rank_position, result in enumerate(search_results, start=1):
        chunk_metadata = result.get("metadata", {})
        typed_meta: ChunkMetadata = {
            "chunk_index": chunk_metadata.get("chunk_index", 0),
            "start_idx": chunk_metadata.get("start_idx", 0),
            "end_idx": chunk_metadata.get("end_idx", 0),
        }
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

    header = (
        f'# Retrieved Context for Query: "{query}"\n'
        f"> Found {len(ranked_results)} relevant chunks with minimum relevance score of {min_score}/10\n"
        "---\n"
    )
    header_tokens = count_tokens(header, model=model)
    remaining_budget = max_tokens - header_tokens
    final_fragments = []
    current_tokens = 0
    for fragment in rag_fragments:
        frag_tokens = count_tokens(fragment, model=model)
        if current_tokens + frag_tokens <= remaining_budget:
            final_fragments.append(fragment)
            current_tokens += frag_tokens
        else:
            console.print(
                f"[yellow]Truncated RAG context to fit {max_tokens} token limit.[/yellow]"
            )
            break

    rag_context = ""
    if final_fragments:
        rag_context = header + "\n---\n".join(final_fragments)
        included_count = len(final_fragments)
        ranked_results = ranked_results[:included_count]

    console.print(
        f"[green]Found {len(ranked_results)} relevant chunks (min_score={min_score})[/green]"
    )
    return rag_context, ranked_results, chunks


if __name__ == "__main__":
    args = get_args()
    output_dir = Path(args.output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[dim]Reading document from: {args.doc_path}[/dim]")
    doc_path = Path(args.doc_path)
    if doc_path.suffix.lower() == ".json":
        with open(doc_path, "r", encoding="utf-8") as f:
            text = json.load(f)
        if not isinstance(text, list):
            raise ValueError(
                f"JSON file must contain a list of strings, got {type(text)}"
            )
    else:
        with open(doc_path, "r", encoding="utf-8") as f:
            text = f.read()

    rag_context, ranked_results, chunks = find_snapshot(
        query=args.query,
        text=text,
        top_k=args.top_k,
        min_score=args.min_score,
        min_results=args.min_results,
        model=args.model,
        criteria=args.criteria,
        include_reasoning=not args.no_reasoning,
        base_url=args.base_url,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        batch_overlap=args.batch_overlap,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        strict_sentences=args.strict_sentences,
        show_progress=not args.no_progress,
        candidate_k=args.candidate_k,
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

    rag_context_tokens = count_tokens(rag_context)
    console.print(
        f"[bold green]✅ Generated {len(chunks)} chunks[/bold green] ([link=file://{chunks_path}]{chunks_path.name}[/link])"
    )
    console.print(
        f"[bold green]✅ Generated {len(ranked_results)} ranked results[/bold green] ([link=file://{ranked_results_path}]{ranked_results_path.name}[/link])"
    )
    console.print(
        f"[bold green]✅ Generated RAG context ({rag_context_tokens} tokens)[/bold green] ([link=file://{rag_context_path}]{rag_context_path.name}[/link])"
    )
