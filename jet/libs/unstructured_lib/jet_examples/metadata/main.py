"""CLI for local unstructured → chroma document ingestion & retrieval."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from jet.libs.unstructured_lib.jet_examples.metadata.retriever import (
    RetrieverConfig,
    UnstructuredLocalRetriever,
)
from rich.console import Console
from rich.logging import RichHandler

console = Console()
# Setup rich logging early

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
    handlers=[RichHandler(console=console, markup=True, rich_tracebacks=True)],
)
logger = logging.getLogger("metadata_retriever")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local document ingestion & semantic retrieval with metadata filtering",
        epilog="Examples are shown at the top of README.md",
    )
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="sentence-transformers model name",
    )
    parser.add_argument(
        "--collection",
        default="local_docs_v1",
        help="Chroma collection name",
    )
    parser.add_argument(
        "--db-dir",
        default="./chroma_db",
        help="where to store ChromaDB data",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── ingest ───────────────────────────────────────────────────────
    ingest = subparsers.add_parser("ingest", help="Ingest file(s) into vector store")
    ingest_group = ingest.add_mutually_exclusive_group(required=True)
    ingest_group.add_argument("--file", type=Path, help="Single file to ingest")
    ingest_group.add_argument("--dir", type=Path, help="Directory to ingest")
    ingest.add_argument(
        "--pattern",
        default="**/*.*",
        help="glob pattern when using --dir (default: **/*.*)",
    )

    # ── query ────────────────────────────────────────────────────────
    query = subparsers.add_parser("query", help="Search documents semantically")
    query.add_argument("question", type=str, help="Search query")
    query.add_argument(
        "--top-k", type=int, default=5, help="number of results (default: 5)"
    )
    query.add_argument(
        "--filter-element-type",
        dest="element_type",
        help="filter by element_type (e.g. Table, NarrativeText, Title, ListItem)",
    )
    query.add_argument(
        "--filter-source-file",
        dest="source_file",
        help="filter by source filename",
    )
    query.add_argument(
        "--filter",
        type=str,
        help='raw JSON filter e.g. \'{"element_type":"Table","source_file":"doc.pdf"}\'',
    )

    return parser.parse_args()


def build_filter(args: argparse.Namespace) -> dict[str, Any] | None:
    """Combine different filter styles into one Chroma-compatible where clause."""
    filters: dict[str, Any] = {}

    if args.element_type:
        filters["element_type"] = args.element_type
    if args.source_file:
        filters["source_file"] = args.source_file
    if args.filter:
        try:
            parsed = json.loads(args.filter)
            if not isinstance(parsed, dict):
                raise ValueError("Filter must be a JSON object")
            filters.update(parsed)
        except Exception as e:
            console.print(f"[bold red]Invalid --filter JSON:[/bold red] {e}")
            sys.exit(1)

    return filters if filters else None


def main() -> None:
    args = parse_args()

    config = RetrieverConfig(
        collection_name=args.collection,
        persist_directory=str(args.db_dir),
        embedding_model_name=args.embedding_model,
    )

    retriever = UnstructuredLocalRetriever(config)

    if args.command == "ingest":
        if args.file:
            try:
                retriever.ingest_file(args.file)
            except Exception:
                logger.exception("Ingestion failed")
                sys.exit(1)
        else:  # --dir
            retriever.ingest_directory(args.dir, pattern=args.pattern)

    elif args.command == "query":
        filters = build_filter(args)
        try:
            result = retriever.retrieve(
                query=args.question,
                top_k=args.top_k,
                filters=filters,
            )
        except Exception:
            logger.exception("Query failed")
            sys.exit(1)

        # ── Pretty print results ─────────────────────────────────────
        if not result["documents"] or not result["documents"][0]:
            console.print("[yellow]No results found.[/yellow]")
            return

        console.rule(f"Top {len(result['documents'][0])} results")

        for i, (doc, meta, dist) in enumerate(
            zip(
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=True,
            ),
            1,
        ):
            source = meta.get("source_file", "unknown")
            el_type = meta.get("element_type", "unknown")
            console.print(f"\n[b]{i}.[/b]  distance = [dim]{dist:.4f}[/dim]")
            console.print(
                f"    [bold cyan]{el_type}[/bold cyan]  •  [blue]{source}[/blue]"
            )
            console.print(
                f"    [white]{doc.strip()[:400]}{'...' if len(doc) > 400 else ''}[/white]"
            )


if __name__ == "__main__":
    main()
