import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from jet.adapters.llama_cpp.chunking_utils import chunk_texts_with_data
from jet.adapters.llama_cpp.config import LLM_MODEL
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem

DEFAULT_FILE_TO_CHUNK = str(
    Path(
        "~/Desktop/External_Projects/AI/repo-libs/smolagents/docs/source/en/guided_tour.md"
    )
    .expanduser()
    .resolve()
)


def _link(path: Path, display_name: str | None = None) -> str:
    """Create a terminal hyperlink with short display name."""
    name = display_name or path.name
    return f"[link=file://{path}]{name}[/link]"


def get_args() -> argparse.Namespace:
    """Parse CLI arguments matching chunk_texts_with_data signature."""
    parser = argparse.ArgumentParser(
        description="Chunk text files with token-aware splitting and rich result display."
    )

    # Positional: multiple files (optional, defaults to single file)
    parser.add_argument(
        "files_to_chunk",
        nargs="*",
        default=None,
        help="One or more text files to chunk (default: guided_tour.md)",
    )

    # Keyword args — 1-2 char shorthands, defaults from chunk_texts_with_data
    parser.add_argument(
        "-s",
        "--chunk-size",
        type=int,
        default=128,
        help="Max tokens per chunk (default: 128)",
    )
    parser.add_argument(
        "-v",
        "--chunk-overlap",
        type=int,
        default=0,
        help="Token overlap between chunks (default: 0)",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=LLM_MODEL,
        help=f"Tokenizer model path/name for token-aware splitting (default: {LLM_MODEL!r} from config, or None = tiktoken)",
    )
    parser.add_argument(
        "-i",
        "--ids",
        nargs="*",
        default=None,
        help="Optional doc IDs, one per input file (default: auto-generated UUIDs)",
    )
    parser.add_argument(
        "-b",
        "--buffer",
        type=int,
        default=0,
        help="Safety buffer subtracted from chunk_size (default: 0)",
    )
    parser.add_argument(
        "-t",
        "--strict-sentences",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
        help="Respect sentence boundaries (default: true)",
    )
    parser.add_argument(
        "-n",
        "--min-chunk-size",
        type=int,
        default=32,
        help="Minimum tokens per chunk to keep (default: 32)",
    )
    parser.add_argument(
        "-p",
        "--show-progress",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
        help="Show progress bar (default: true)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=str(OUTPUT_DIR),
        type=str,
        help=f"Output directory (default: '{OUTPUT_DIR}')",
    )

    return parser.parse_args()


def _display_chunk_summary(
    chunks: list[dict[str, Any]],
    file_paths: list[str],
    args: argparse.Namespace,
) -> None:
    """Display rich-formatted summary of chunking results."""
    if not chunks:
        console.print(
            Panel(
                Text(
                    "No chunks generated — check input files and parameters.",
                    style="yellow",
                ),
                border_style="yellow",
                title="Chunking Results",
            )
        )
        return

    # Per-file statistics
    file_stats: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        doc_id = chunk["doc_id"]
        if doc_id not in file_stats:
            file_stats[doc_id] = {
                "file": file_paths[chunk["doc_index"]]
                if chunk["doc_index"] < len(file_paths)
                else "unknown",
                "chunk_count": 0,
                "total_tokens": 0,
                "avg_tokens": 0.0,
                "min_tokens": float("inf"),
                "max_tokens": 0,
            }
        stats = file_stats[doc_id]
        stats["chunk_count"] += 1
        tokens = chunk["num_tokens"]
        stats["total_tokens"] += tokens
        stats["min_tokens"] = min(stats["min_tokens"], tokens)
        stats["max_tokens"] = max(stats["max_tokens"], tokens)

    for stats in file_stats.values():
        if stats["chunk_count"] > 0:
            stats["avg_tokens"] = stats["total_tokens"] / stats["chunk_count"]

    # Header
    console.print()
    console.print(
        Panel(
            Text("Text Chunking Results", style="bold white on green"),
            border_style="green",
            padding=(0, 2),
        )
    )

    # Parameters table
    params_table = Table(
        title="Chunking Parameters",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    params_table.add_column("Parameter", style="bold", width=22)
    params_table.add_column("Value", width=30)
    params_table.add_row("Input files", str(len(file_paths)))
    params_table.add_row("Chunk size", str(args.chunk_size))
    params_table.add_row("Chunk overlap", str(args.chunk_overlap))
    params_table.add_row("Model", str(args.model))
    params_table.add_row("Buffer", str(args.buffer))
    params_table.add_row("Strict sentences", str(args.strict_sentences))
    params_table.add_row("Min chunk size", str(args.min_chunk_size))
    params_table.add_row("Total chunks", str(len(chunks)))
    console.print(params_table)

    # Per-file breakdown
    file_table = Table(
        title="Per-File Breakdown",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    file_table.add_column("File", style="bold", max_width=40)
    file_table.add_column("Chunks", justify="right")
    file_table.add_column("Total Tokens", justify="right")
    file_table.add_column("Avg Tokens", justify="right")
    file_table.add_column("Min Tokens", justify="right")
    file_table.add_column("Max Tokens", justify="right")

    for doc_id, stats in file_stats.items():
        file_name = Path(stats["file"]).name
        file_table.add_row(
            file_name,
            str(stats["chunk_count"]),
            str(stats["total_tokens"]),
            f"{stats['avg_tokens']:.1f}",
            str(stats["min_tokens"]),
            str(stats["max_tokens"]),
        )

    console.print(file_table)

    # Chunk preview (first 5)
    preview_table = Table(
        title=f"Chunk Preview (first 5 of {len(chunks)})",
        box=box.SIMPLE,
        show_header=True,
        header_style="bold green",
    )
    preview_table.add_column("ID", style="dim", max_width=10)
    preview_table.add_column("Doc", justify="center", width=4)
    preview_table.add_column("Chunk#", justify="right", width=7)
    preview_table.add_column("Tokens", justify="right", width=7)
    preview_table.add_column("Content Preview", max_width=60)

    for chunk in chunks[:5]:
        chunk_id_short = chunk["id"][:8]
        preview = chunk["content"].replace("\n", " ")[:80]
        if len(chunk["content"]) > 80:
            preview += "…"
        preview_table.add_row(
            chunk_id_short,
            str(chunk["doc_index"]),
            str(chunk["chunk_index"]),
            str(chunk["num_tokens"]),
            preview,
        )

    console.print(preview_table)
    console.print()


def main() -> None:
    """Load multiple files, chunk them, save results, and display summary."""
    args = get_args()

    output_dir = Path(args.output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Apply default file if none provided
    files_to_chunk = (
        args.files_to_chunk if args.files_to_chunk else [DEFAULT_FILE_TO_CHUNK]
    )

    console.print(
        f"[bold]Output directory:[/bold] {_link(output_dir, 'chunks_output')}"
    )
    console.print(f"[bold]Files to process:[/bold] {len(files_to_chunk)}")
    for f in files_to_chunk:
        console.print(f"  • {Path(f).name}")

    # Read all file contents
    texts: list[str] = []
    valid_file_paths: list[str] = []
    for file_path in files_to_chunk:
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            console.print(f"[yellow]⚠ File not found, skipping:[/yellow] {path}")
            continue
        try:
            content = path.read_text(encoding="utf-8")
            texts.append(content)
            valid_file_paths.append(str(path))
            console.print(
                f"[green]✓ Loaded:[/green] {path.name} ({len(content):,} chars)"
            )
        except Exception as e:
            console.print(f"[red]✗ Failed to read {path.name}:[/red] {e}")

    if not texts:
        console.print("[red]No valid files to process. Exiting.[/red]")
        return

    # Chunk all texts
    console.print()
    console.print("[bold]Chunking in progress...[/bold]")
    chunks = chunk_texts_with_data(
        texts=texts,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        model=args.model,
        ids=args.ids,
        buffer=args.buffer,
        strict_sentences=args.strict_sentences,
        min_chunk_size=args.min_chunk_size,
        show_progress=args.show_progress,
    )

    # Save results
    chunks_file = output_dir / "chunks.json"
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    console.print(
        f"[green]✓ Saved {len(chunks)} chunks:[/green] {_link(chunks_file, 'chunks.json')}"
    )

    # Save parameters for reproducibility
    params_file = output_dir / "params.json"
    params = {
        "files": valid_file_paths,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "model": args.model,
        "buffer": args.buffer,
        "strict_sentences": args.strict_sentences,
        "min_chunk_size": args.min_chunk_size,
        "ids": args.ids,
    }
    with open(params_file, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    console.print(
        f"[green]✓ Saved parameters:[/green] {_link(params_file, 'params.json')}"
    )

    # Display results
    _display_chunk_summary(chunks, valid_file_paths, args)


if __name__ == "__main__":
    main()
