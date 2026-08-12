"""Demo: Semantic Indexing

Optimized for embedding pipelines. Drops URLs (noise), keeps anchor/alt text,
and applies markdown chunking at h3 boundaries.
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

TARGET_URL = "https://example.com/article"
BASE_API = os.environ.get("JINA_READER_BASE_URL", "http://localhost:3001").rstrip("/")


def save_json(path: Path, data: str | dict) -> None:
    """Parse and pretty-print JSON before saving for readability."""
    try:
        parsed = json.loads(data) if isinstance(data, str) else data
        path.write_text(
            json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except (json.JSONDecodeError, TypeError):
        path.write_text(data if isinstance(data, str) else str(data), encoding="utf-8")


def run_indexing_demo():
    console.print("[bold cyan]Fetching article for semantic indexing...[/]")
    resp = requests.get(
        f"{BASE_API}/{TARGET_URL}",
        headers={
            "Accept": "application/json",
            "x-retain-links": "text",
            "x-retain-images": "alt",
            "x-markdown-chunking": "h3",
        },
        timeout=60,
    )
    out_file = OUTPUT_DIR / "semantic_chunks.json"
    save_json(out_file, resp.text)

    console.print("\n[bold green]✔ Saved results:[/]")
    console.print(f"  📄 [link=file://{out_file}]{out_file.name}[/link]")


if __name__ == "__main__":
    run_indexing_demo()
