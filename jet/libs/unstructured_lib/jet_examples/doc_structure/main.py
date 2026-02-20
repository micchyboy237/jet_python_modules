#!/usr/bin/env python3
"""
CLI tool to prepare documents for RAG using unstructured + rag_unstructured_local.py

Reuses the core processing functions from rag_unstructured_local.

Examples:
  python unstructured_rag_cli.py --file invoice.pdf --output chunks/
  python unstructured_rag_cli.py --dir docs/ --output processed/ --chunk-size 350
  python unstructured_rag_cli.py --dir contracts --output out --strategy hi_res --verbose
"""

import argparse
import sys
from pathlib import Path

try:
    from .rag_unstructured_local import (
        SUPPORTED_EXTENSIONS,
        process_directory,
        process_document,
    )
except ImportError:
    print("Error: Could not import rag_unstructured_local", file=sys.stderr)
    print(
        "Make sure the file is in the same directory or in PYTHONPATH.", file=sys.stderr
    )
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare local documents for LLM RAG using unstructured",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single file:
    python unstructured_rag_cli.py --file report.pdf -o chunks/

  Directory (recursive):
    python unstructured_rag_cli.py --dir documents/ --output processed/ --chunk-size 400

  Advanced:
    python unstructured_rag_cli.py --dir ./docs --output ./rag-ready --strategy hi_res --verbose
""",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=Path, help="Path to a single document")
    group.add_argument("--dir", type=Path, help="Directory to process recursively")

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output directory for .jsonl chunk files",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=450,
        help="Approximate max tokens per chunk (default: 450)",
    )
    parser.add_argument(
        "--strategy",
        choices=["auto", "fast", "hi_res", "ocr_only"],
        default="auto",
        help="Partitioning strategy passed to unstructured (default: auto)",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Skip including metadata in output chunks",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show progress and more detailed messages",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    args.output.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print(f"Mode: {'Single file' if args.file else 'Directory'}")
        print(f"Output: {args.output.resolve()}")
        print(f"Chunk size: ~{args.chunk_size} tokens")
        print(f"Strategy: {args.strategy}")
        print(f"Metadata: {'included' if not args.no_metadata else 'excluded'}")
        print("-" * 60)

    try:
        if args.file:
            if not args.file.is_file():
                parser.error(f"Not a file: {args.file}")

            if args.verbose:
                print(f"Processing single file: {args.file}")

            # For single file we use the reusable function directly
            chunks = process_document(
                filepath=args.file,
                chunk_size=args.chunk_size,
                include_metadata=not args.no_metadata,
            )

            # Write result (mimicking what process_directory does)
            out_path = args.output / f"{args.file.stem}.jsonl"
            with out_path.open("w", encoding="utf-8") as f:
                for chunk in chunks:
                    f.write(
                        chunk.json() + "\n"
                    )  # assuming dict has .json() or use json.dumps

            print(f"✓ {args.file.name} → {len(chunks)} chunks → {out_path}")

        elif args.dir:
            if not args.dir.is_dir():
                parser.error(f"Not a directory: {args.dir}")

            if args.verbose:
                print(f"Processing directory: {args.dir.resolve()}")

            # Reuse the batch function — it already handles writing JSONL files
            process_directory(
                input_dir=args.dir,
                output_dir=args.output,
                chunk_size=args.chunk_size,
            )

            print(f"Directory processing complete. Output in: {args.output}")

    except Exception as e:
        print(f"Error during processing: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
