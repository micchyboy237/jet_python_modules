"""Demo: Geo- and Locale-Sensitive Scraping

Pins geography, language, and cookies for region-gated content.
Requires a premium Jina API key.
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

TARGET_URL = "https://books.toscrape.com"
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


def run_geo_demo():
    console.print("[bold cyan]Fetching with German geo/locale proxy...[/]")
    resp = requests.get(
        f"{BASE_API}/{TARGET_URL}",
        headers={
            "Accept": "application/json",
            "x-proxy": "de",
            "x-locale": "de-DE",
            "x-set-cookie": "country=DE; Path=/",
        },
        timeout=60,
    )
    out_file = OUTPUT_DIR / "geo_de_product.json"
    save_json(out_file, resp.text)

    console.print("\n[bold green]✔ Saved results:[/]")
    console.print(f"  📄 [link=file://{out_file}]{out_file.name}[/link]")


if __name__ == "__main__":
    run_geo_demo()
