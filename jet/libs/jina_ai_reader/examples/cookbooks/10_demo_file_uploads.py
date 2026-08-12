"""Demo: PDF and MS Office File Uploads

Ingests binary files (PDF, DOCX, XLSX, PPTX) via multipart upload.
Reader sniffs MIME type from bytes. Supports page selection for PDFs.
"""

import json
import os
import shutil
from pathlib import Path

import requests
from rich.console import Console

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_API = os.environ.get("JINA_READER_BASE_URL", "http://localhost:3001").rstrip("/")

# Path to the real PDF used for this demo
SOURCE_PDF = Path(
    "/Users/jethroestrada/Desktop/External_Projects/AI/examples/RAG_Techniques/data/Understanding_Climate_Change.pdf"
)


def save_json(path: Path, data: str | dict) -> None:
    """Parse and pretty-print JSON before saving for readability."""
    try:
        parsed = json.loads(data) if isinstance(data, str) else data
        path.write_text(
            json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except (json.JSONDecodeError, TypeError):
        path.write_text(data if isinstance(data, str) else str(data), encoding="utf-8")


def run_file_upload_demo():
    saved_files = []

    # Verify the source PDF exists before doing any uploads
    if not SOURCE_PDF.exists():
        console.print(f"[bold red]✘ Source PDF not found:[/] {SOURCE_PDF}")
        return

    console.print(f"[bold cyan]Using PDF:[/] {SOURCE_PDF.name}")

    # 1. Full PDF upload with chunking
    console.print("[bold cyan]1. Uploading PDF with s3 chunking...[/]")
    with open(SOURCE_PDF, "rb") as f:
        resp_full = requests.post(
            f"{BASE_API}/",
            files={"file": (SOURCE_PDF.name, f, "application/pdf")},
            headers={"Accept": "application/json", "x-markdown-chunking": "s3"},
            timeout=60,
        )
    out_full = OUTPUT_DIR / "pdf_full_upload.json"
    save_json(out_full, resp_full.text)
    saved_files.append(out_full)

    # 2. Single PDF page extraction
    console.print("[bold cyan]2. Extracting single page from PDF...[/]")
    with open(SOURCE_PDF, "rb") as f:
        resp_page = requests.post(
            f"{BASE_API}/",
            files={"file": (SOURCE_PDF.name, f, "application/pdf")},
            data={"page": "1"},
            headers={"Accept": "application/json"},
            timeout=60,
        )
    out_page = OUTPUT_DIR / "pdf_page_1.json"
    save_json(out_page, resp_page.text)
    saved_files.append(out_page)

    console.print("\n[bold green]✔ Saved results:[/]")
    for f in saved_files:
        console.print(f"  📄 [link=file://{f}]{f.name}[/link]")


if __name__ == "__main__":
    run_file_upload_demo()
