# main.py

import argparse
import shutil
from pathlib import Path

from jet.file.utils import save_file
from jet.libs.unstructured_lib.jet_examples.rag_grok_agents_answer.rag_pipeline import (
    RAGPipeline,
)
from rich.console import Console

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

console = Console()


def get_args() -> argparse.Namespace:
    """
    Parse CLI arguments.

    query and file_paths_or_dir:
    - Required
    - Can be positional OR via flags
    """

    parser = argparse.ArgumentParser(description="Run RAG pipeline ingestion + query.")

    # Positional arguments (optional at parser level, validated after)
    parser.add_argument(
        "query_positional",
        nargs="?",
        help="Query string (positional alternative to -q/--query).",
    )

    parser.add_argument(
        "path_positional",
        nargs="?",
        help="File path or directory (positional alternative to -p/--path).",
    )

    # Optional named arguments (required=False here, validated manually)
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        help="Query string.",
    )

    parser.add_argument(
        "-p",
        "--path",
        type=str,
        help="File path or directory to ingest.",
    )

    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=5,
        help="Number of top documents to retrieve (default: 5).",
    )

    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature (default: 0.0).",
    )

    args = parser.parse_args()

    # Resolve query
    resolved_query = args.query or args.query_positional
    resolved_path = args.path or args.path_positional

    if not resolved_query:
        parser.error("Query is required (use positional or -q/--query).")

    if not resolved_path:
        parser.error(
            "File path or directory is required (use positional or -p/--path)."
        )

    args.query = resolved_query
    args.file_paths_or_dir = resolved_path

    return args


def main() -> None:
    args = get_args()

    pipeline = RAGPipeline()

    console.rule("Starting ingestion", style="bold blue")
    console.print(f"[cyan]Ingesting from:[/cyan] {args.file_paths_or_dir}")

    pipeline.ingest(args.file_paths_or_dir)

    console.rule("Query", style="bold magenta")
    console.print(f"[yellow]Running query:[/yellow] {args.query}")
    console.print(
        f"[dim]Top-K:[/dim] {args.top_k} | [dim]Temperature:[/dim] {args.temperature}"
    )

    answer = pipeline.query(
        args.query,
        k=args.top_k,
        temperature=args.temperature,
    )

    console.rule("Result", style="bold green")
    console.print("[bold green]Answer:[/bold green]")
    console.print(answer, markup=False)

    save_file(
        {
            "query": args.query,
            "file_paths_or_dir": args.file_paths_or_dir,
            "top_k": args.top_k,
            "temperature": args.temperature,
        },
        OUTPUT_DIR / "input.json",
    )

    save_file(answer, OUTPUT_DIR / "answer.md")


if __name__ == "__main__":
    main()
