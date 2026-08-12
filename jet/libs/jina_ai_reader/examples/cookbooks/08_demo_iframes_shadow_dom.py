"""Demo: Iframes and Shadow DOM

Extracts content from embedded iframes and shadow roots that are normally
skipped during serialization. Essential for web-component-heavy sites.
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

TARGET_URL = "ttps://github.com/jina-ai/reader"
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


def run_iframe_shadow_demo():
    console.print("[bold cyan]Extracting iframe + shadow DOM content...[/]")
    resp = requests.get(
        f"{BASE_API}/{TARGET_URL}",
        headers={
            "Accept": "application/json",
            "x-with-iframe": "true",
            "x-with-shadow-dom": "true",
            "x-timeout": "60",
        },
        timeout=120,
    )
    out_file = OUTPUT_DIR / "iframe_shadow_content.json"
    save_json(out_file, resp.text)

    console.print("\n[bold green]✔ Saved results:[/]")
    console.print(f"  📄 [link=file://{out_file}]{out_file.name}[/link]")


if __name__ == "__main__":
    run_iframe_shadow_demo()
