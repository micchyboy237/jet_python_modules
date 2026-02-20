"""
Minimal RAG-prep pipeline using unstructured + local files.
No LangChain, no llama_index, no external APIs.

Features shown:
- Partition many file types automatically
- Clean + chunk by semantic sections (titles)
- Produce JSONL ready for vector DB / embedding
- Basic metadata filtering
"""

import json
import re
from pathlib import Path

from unstructured.chunking.title import chunk_by_title
from unstructured.cleaners.core import clean_non_ascii_chars
from unstructured.partition.auto import partition

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


def process_document(
    filepath: str | Path,
    chunk_size: int = 400,
    include_metadata: bool = True,
    strategy: str = "auto",  # allow "hi_res", "fast", "ocr_only" for pdf/images
    partition_kwargs: dict | None = None,
    clean_whitespace: bool = True,  # ← NEW: toggle whitespace cleaning
    combine_under_chars: int = 100,  # ← NEW: make tunable
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
    strategy: str = "auto",  # added for consistency
) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

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
            except Exception as exc:
                print(f"Failed {path.name}: {exc}")


if __name__ == "__main__":
    single_file = "example_docs/contract.pdf"
    chunks = process_document(single_file, chunk_size=350, strategy="hi_res")
    print(f"Got {len(chunks)} chunks from {single_file}")
    print(json.dumps(chunks[:2], indent=2))
