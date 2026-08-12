"""Demo: Deep Research

For long-context models. Keeps anchor text inline but deduplicates URLs
into a single footer summary to avoid token bloat from repeated links.
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

TARGET_URL = "https://en.wikipedia.org/wiki/Artificial_intelligence"
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


def run_research_demo():
    console.print("[bold cyan]Fetching article for deep research agent...[/]")
    resp = requests.get(
        f"{BASE_API}/{TARGET_URL}",
        headers={
            "Accept": "application/json",
            "x-retain-links": "text",
            "x-with-links-summary": "true",
            "x-retain-images": "alt",
        },
        timeout=60,
    )
    out_file = OUTPUT_DIR / "deep_research.json"
    save_json(out_file, resp.text)

    console.print("\n[bold green]✔ Saved results:[/]")
    console.print(f"  📄 [link=file://{out_file}]{out_file.name}[/link]")


if __name__ == "__main__":
    run_research_demo()
