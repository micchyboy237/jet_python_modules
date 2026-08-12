"""Demo: Visual Snapshot / Pageshot

Captures the full rendered page as an image for multimodal reasoning or QA.
Removes overlays and waits for media to settle.
"""

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


def run_pageshot_demo():
    console.print("[bold cyan]Capturing full-page pageshot...[/]")
    resp = requests.get(
        f"{BASE_API}/{TARGET_URL}",
        headers={
            "x-respond-with": "pageshot",
            "x-remove-overlay": "true",
            "x-timeout": "30",
        },
        timeout=60,
    )
    # Pageshot returns binary image data, not JSON
    out_file = OUTPUT_DIR / "pageshot.png"
    out_file.write_bytes(resp.content)

    console.print("\n[bold green]✔ Saved results:[/]")
    console.print(f"  🖼️  [link=file://{out_file}]{out_file.name}[/link]")


if __name__ == "__main__":
    run_pageshot_demo()
