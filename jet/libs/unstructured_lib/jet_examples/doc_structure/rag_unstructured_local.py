"""
Minimal RAG-prep pipeline using unstructured + local files.
No LangChain, no llama_index, no external APIs.

Features shown:
- Partition many file types automatically
- Clean + chunk by semantic sections (titles)
- Produce JSONL ready for vector DB / embedding
- Basic metadata filtering
"""

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Literal

from unstructured.chunking.title import chunk_by_title
from unstructured.cleaners.core import clean_non_ascii_chars
from unstructured.partition.auto import partition

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Comprehensive list based on unstructured docs[](https://docs.unstructured.io/open-source/core-functionality/partitioning)
# Covers auto-detected / routed formats; some require extras (libreoffice, pandoc, tesseract, unstructured[pdf], etc.)
SUPPORTED_EXTENSIONS = {
    # Core text & markup
    ".txt",
    ".text",
    ".log",
    ".md",
    ".markdown",
    ".html",
    ".htm",
    ".xml",
    ".json",
    # Structured data
    ".csv",
    ".tsv",
    # Office & documents
    ".pdf",
    ".docx",
    ".doc",
    ".odt",
    ".rtf",
    ".epub",
    ".org",
    ".rst",
    # Presentations
    ".pptx",
    ".ppt",
    # Spreadsheets
    ".xlsx",
    ".xls",
    # Emails
    ".eml",
    ".msg",
    # Images (OCR/layout needed)
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".tif",
    ".bmp",
    ".heic",
}

StrategyType = Literal["auto", "fast", "hi_res", "ocr_only"]


def process_document(
    filepath: str | Path,
    chunk_size: int = 400,
    include_metadata: bool = True,
    strategy: str = "auto",  # allow "hi_res", "fast", "ocr_only" for pdf/images
    partition_kwargs: dict | None = None,
    clean_whitespace: bool = True,
    combine_under_chars: int = 100,
) -> list[dict]:
    path = Path(filepath)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported extension: {ext}")

    elements = partition(
        filename=str(path),
        strategy=strategy,
        infer_table_structure=True,
        languages=["eng"],
        **(partition_kwargs or {}),
    )

    # Mild whitespace cleaning: preserve newlines, collapse horizontal whitespace
    def mild_clean_whitespace(text: str) -> str:
        if not text:
            return text
        # Collapse multiple spaces/tabs → single space
        text = re.sub(r"[ \t]+", " ", text)
        # Normalize multiple newlines → double newline (paragraph break)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        # Optional: strip leading/trailing whitespace per line
        text = "\n".join(line.strip() for line in text.splitlines())
        return text.strip()

    # Apply cleaning only if requested
    if clean_whitespace:
        for el in elements:
            if hasattr(el, "text") and el.text:
                el.text = mild_clean_whitespace(el.text)
                el.text = clean_non_ascii_chars(el.text)

    # Semantic chunking
    chunks = chunk_by_title(
        elements,
        max_characters=chunk_size * 4,  # rough tokens → chars
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
                # Optional: add .coordinates, .parent_id, .category_depth if useful later
            }
        records.append(record)

    return records


def process_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    chunk_size: int = 400,
    strategy: str = "auto",
) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    processed_count = 0
    for path in input_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
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
                print(f"Processed {path.name} → {len(records)} chunks")
                processed_count += 1
            except Exception as exc:
                print(f"Failed {path.name}: {exc}")
    print(f"Directory processing complete. {processed_count} file(s) processed.")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal RAG preprocessing pipeline: partition → clean → chunk → JSONL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "input",
        help="Input file or directory",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=OUTPUT_DIR,
        help="Output **directory** for JSONL files",
    )
    parser.add_argument(
        "-c",
        "--chunk-size",
        type=int,
        default=400,
        help="Approximate maximum chunk size in characters",
    )
    parser.add_argument(
        "-s",
        "--strategy",
        choices=["auto", "fast", "hi_res", "ocr_only"],
        default="auto",
        help="Partitioning strategy (especially relevant for PDFs/images)",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Do not include metadata in output JSON",
    )
    parser.add_argument(
        "--no-clean-whitespace",
        action="store_true",
        help="Skip mild whitespace normalization",
    )
    parser.add_argument(
        "--combine-under",
        type=int,
        default=100,
        help="Combine small sections under this char count",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    input_path = Path(args.input)
    output_dir = Path(args.output) if args.output is not None else None

    include_metadata = not args.no_metadata
    clean_ws = not args.no_clean_whitespace

    # Create output directory only if we're actually going to write files
    should_write = True  # will be adjusted below for single-file no-output case

    if input_path.is_file():
        try:
            chunks = process_document(
                input_path,
                chunk_size=args.chunk_size,
                include_metadata=include_metadata,
                strategy=args.strategy,  # type: ignore[arg-type]
                clean_whitespace=clean_ws,
                combine_under_chars=args.combine_under,
            )  # type: ignore

            # Decide whether to write or preview
            explicit_output = args.output is not None
            should_write = explicit_output

            if should_write:
                output_dir.mkdir(parents=True, exist_ok=True)
                # Always save as <stem>.jsonl in output directory
                out_filename = f"{input_path.stem}.jsonl"
                out_path = output_dir / out_filename
                with out_path.open("w", encoding="utf-8") as f:
                    for chunk in chunks:
                        f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                print(f"Wrote {len(chunks)} chunks → {out_path}")
            else:
                # Print preview
                print(f"Extracted {len(chunks)} chunks from {input_path.name}")
                if chunks:
                    print(json.dumps(chunks[:3], indent=2, ensure_ascii=False))
                else:
                    print("No chunks produced.")

        except Exception as exc:
            print(f"Error processing file {input_path}: {exc}")

    elif input_path.is_dir():
        if args.output is None:
            parser = argparse.ArgumentParser()
            parser.error("--output directory is required when INPUT is a directory")
        output_dir.mkdir(parents=True, exist_ok=True)

        process_directory(
            input_path,
            output_dir,
            chunk_size=args.chunk_size,
            strategy=args.strategy,
        )

    else:
        parser = argparse.ArgumentParser()
        parser.error("Input must be an existing file or directory")
