"""
Minimal RAG-prep pipeline using unstructured + local files.
No LangChain, no llama_index, no external APIs.

Features shown:
- Partition many file types automatically
- Clean + chunk by semantic sections (titles)
- Produce JSONL ready for vector DB / embedding
- Basic metadata filtering
- Rich logging + progress tracking
"""

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Literal, Set

from rich.console import Console
from rich.table import Table
from rich.traceback import install
from tqdm import tqdm
from unstructured.chunking.title import chunk_by_title
from unstructured.cleaners.core import clean_non_ascii_chars
from unstructured.partition.auto import partition

install()
console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SUPPORTED_EXTENSIONS: Set[str] = {
    ".txt",
    ".text",
    ".log",
    ".md",
    ".markdown",
    ".html",
    ".htm",
    ".xml",
    ".json",
    ".csv",
    ".tsv",
    ".pdf",
    ".docx",
    ".doc",
    ".odt",
    ".rtf",
    ".epub",
    ".org",
    ".rst",
    ".pptx",
    ".ppt",
    ".xlsx",
    ".xls",
    ".eml",
    ".msg",
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".tif",
    ".bmp",
    ".heic",
}


def normalize_extension(ext: str) -> str:
    ext = ext.strip().lower()
    if not ext.startswith("."):
        ext = "." + ext
    return ext


StrategyType = Literal["auto", "fast", "hi_res", "ocr_only"]


# ---------------------------
# Core Processing
# ---------------------------


def process_document(
    filepath: str | Path,
    chunk_size: int = 400,
    include_metadata: bool = True,
    strategy: StrategyType = "auto",
    partition_kwargs: dict | None = None,
    clean_whitespace: bool = True,
    combine_under_chars: int = 100,
) -> list[dict]:
    path = Path(filepath)

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    # We'll pass allowed_extensions from caller in directory mode
    # For single file we still use default check (can be relaxed later if needed)
    if ext not in DEFAULT_SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported extension: {ext}")

    console.log(f"[cyan]Partitioning[/cyan] {path.name}")

    elements = partition(
        filename=str(path),
        strategy=strategy,
        infer_table_structure=True,
        languages=["eng"],
        **(partition_kwargs or {}),
    )

    def mild_clean_whitespace(text: str) -> str:
        if not text:
            return text
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        text = "\n".join(line.strip() for line in text.splitlines())
        return text.strip()

    if clean_whitespace:
        for el in elements:
            if hasattr(el, "text") and el.text:
                el.text = mild_clean_whitespace(el.text)
                el.text = clean_non_ascii_chars(el.text)

    chunks = chunk_by_title(
        elements,
        max_characters=chunk_size * 4,
        new_after_n_chars=chunk_size * 4,
        combine_text_under_n_chars=combine_under_chars,
        multipage_sections=True,
    )

    records = []

    for chunk in chunks:
        if not chunk.text.strip():
            continue

        record = {
            "text": chunk.text.strip(),
            "type": chunk.category,
            "source_file": path.name,
        }

        if include_metadata:
            record["metadata"] = {
                "page_number": chunk.metadata.page_number,
                "filename": chunk.metadata.filename,
            }

        records.append(record)

    console.log(f"[green]✔ Extracted[/green] {len(records)} chunks from {path.name}")

    return records


# ---------------------------
# Directory Processing
# ---------------------------


def process_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    chunk_size: int = 400,
    strategy: StrategyType = "auto",
    allowed_extensions: Set[str] | None = None,
) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    exts = allowed_extensions or DEFAULT_SUPPORTED_EXTENSIONS

    all_files = [
        p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts
    ]

    if not all_files:
        console.print("[yellow]⚠ No supported files found.[/yellow]")
        return

    console.print(f"[bold cyan]Found {len(all_files)} supported file(s).[/bold cyan]")

    processed = 0
    failed = []

    for path in tqdm(all_files, desc="Processing files"):
        try:
            records = process_document(
                path,
                chunk_size=chunk_size,
                strategy=strategy,
            )

            out_path = output_dir / f"{path.stem}.jsonl"

            with out_path.open("w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            processed += 1

        except Exception as exc:
            failed.append((path.name, str(exc)))
            console.print(f"[red]✖ Failed[/red] {path.name}: {exc}")

    # Summary table
    table = Table(title="Directory Processing Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Files", str(len(all_files)))
    table.add_row("Processed", str(processed))
    table.add_row("Failed", str(len(failed)))

    console.print(table)

    if failed:
        console.print("\n[bold red]Failed Files:[/bold red]")
        for name, err in failed:
            console.print(f"  • {name} → {err}")


# ---------------------------
# CLI
# ---------------------------


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal RAG preprocessing pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("input", help="Input file or directory")

    parser.add_argument(
        "-o",
        "--output",
        default=OUTPUT_DIR,
        help="Output directory for JSONL files",
    )

    parser.add_argument(
        "-c",
        "--chunk-size",
        type=int,
        default=400,
        help="Approximate maximum chunk size",
    )

    parser.add_argument(
        "-s",
        "--strategy",
        choices=["auto", "fast", "hi_res", "ocr_only"],
        default="auto",
    )

    parser.add_argument("--no-metadata", action="store_true")
    parser.add_argument("--no-clean-whitespace", action="store_true")

    parser.add_argument(
        "--combine-under",
        type=int,
        default=100,
    )

    parser.add_argument(
        "-e",
        "--extensions",
        type=str,
        default=None,
        help="Comma-separated list of file extensions to process (e.g. '.pdf,.md,txt') — overrides default set",
    )

    return parser.parse_args()


# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    args = get_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    include_metadata = not args.no_metadata
    clean_ws = not args.no_clean_whitespace

    # Prepare allowed extensions
    allowed_exts: Set[str] | None = None
    if args.extensions:
        raw = [e.strip() for e in args.extensions.split(",") if e.strip()]
        allowed_exts = {normalize_extension(e) for e in raw}
        console.print(
            f"[dim]Using custom extensions filter: {sorted(allowed_exts)}[/dim]"
        )

    if input_path.is_file():
        try:
            # For single file: if user gave --extensions, we can skip strict check
            # but for simplicity we still use default set unless very restrictive
            # (you can remove the check entirely if preferred)
            chunks = process_document(
                input_path,
                chunk_size=args.chunk_size,
                include_metadata=include_metadata,
                strategy=args.strategy,
                clean_whitespace=clean_ws,
                combine_under_chars=args.combine_under,
            )

            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / f"{input_path.stem}.jsonl"

            with out_path.open("w", encoding="utf-8") as f:
                for chunk in tqdm(chunks, desc="Writing chunks"):
                    f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

            console.print(
                f"[bold green]✔ Wrote {len(chunks)} chunks → {out_path}[/bold green]"
            )

        except Exception as exc:
            console.print(f"[bold red]Error:[/bold red] {exc}")

    elif input_path.is_dir():
        process_directory(
            input_path,
            output_dir,
            chunk_size=args.chunk_size,
            strategy=args.strategy,
            allowed_extensions=allowed_exts,
        )

    else:
        console.print(
            "[bold red]Input must be an existing file or directory.[/bold red]"
        )
