#!/usr/bin/env python3
"""
CLI tool to prepare documents for RAG using unstructured + rag_unstructured_local.py

Reuses the core processing functions from rag_unstructured_local.

Examples:
  Single file (short form):
    python unstructured_rag_cli.py -f invoice.pdf
    python unstructured_rag_cli.py -f report.pdf -o chunks/

  Directory (short form):
    python unstructured_rag_cli.py -d docs/
    python unstructured_rag_cli.py -d contracts -o processed/ --chunk-size 350

  Advanced:
    python unstructured_rag_cli.py -d ./docs --strategy hi_res --verbose
"""

import argparse
import json
import sys
from pathlib import Path

# Adjust import path according to your actual project structure
try:
    from jet.libs.unstructured_lib.jet_examples.doc_structure.rag_unstructured_local import (
        process_directory,
        process_document,
    )
except ImportError:
    print("Error: Could not import rag_unstructured_local", file=sys.stderr)
    print("Check PYTHONPATH or file location.", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare local documents for LLM RAG using unstructured",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single file:
    python unstructured_rag_cli.py -f report.pdf
    python unstructured_rag_cli.py -f invoice.pdf -o chunks/

  Directory:
    python unstructured_rag_cli.py -d documents/
    python unstructured_rag_cli.py -d ./docs --output rag-ready --chunk-size 400

  Advanced:
    python unstructured_rag_cli.py -d contracts --strategy hi_res -v
""",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-f", "--file", type=Path, help="Path to a single document to process"
    )
    group.add_argument(
        "-d", "--dir", type=Path, help="Directory to process recursively"
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("."),
        help="Output directory for .jsonl files (default: current directory '.')",
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
        help="Partitioning strategy (default: auto)",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Do not include metadata in output chunks",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed progress messages"
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    args.output.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print(f"Output directory: {args.output.resolve()}")
        print(f"Chunk size: ~{args.chunk_size} tokens")
        print(f"Strategy: {args.strategy}")
        print(f"Metadata: {'included' if not args.no_metadata else 'excluded'}")
        print("-" * 60)

    try:
        if args.file:
            if not args.file.is_file():
                parser.error(f"Not a valid file: {args.file}")

            if args.verbose:
                print(f"Processing single file: {args.file}")

            chunks = process_document(
                filepath=args.file,
                chunk_size=args.chunk_size,
                include_metadata=not args.no_metadata,
                # strategy=args.strategy,   # uncomment if your process_document supports it
            )

            out_path = args.output / f"{args.file.stem}.jsonl"
            with out_path.open("w", encoding="utf-8") as f:
                for chunk in chunks:
                    json.dump(chunk, f, ensure_ascii=False)
                    f.write("\n")

            print(f"✓ {args.file.name} → {len(chunks)} chunks → {out_path}")

        elif args.dir:
            if not args.dir.is_dir():
                parser.error(f"Not a valid directory: {args.dir}")

            if args.verbose:
                print(f"Processing directory: {args.dir.resolve()}")

            process_directory(
                input_dir=args.dir,
                output_dir=args.output,
                chunk_size=args.chunk_size,
                # strategy=args.strategy,   # uncomment if supported
            )

            print(f"Directory processing complete. Files written to: {args.output}")

    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
