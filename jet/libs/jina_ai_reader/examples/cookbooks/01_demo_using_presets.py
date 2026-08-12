"""Demo: Using Presets

Demonstrates the x-preset header shortcut that bundles common options.
Shows both a pure preset call and an override scenario.
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


def run_preset_demo():
    # 1. Pure preset usage (index)
    console.print("[bold cyan]1. Fetching with 'index' preset...[/]")
    resp_index = requests.get(
        f"{BASE_API}/{TARGET_URL}",
        headers={"x-preset": "index", "Accept": "application/json"},
        timeout=60,
    )
    out_index = OUTPUT_DIR / "preset_index.json"
    save_json(out_index, resp_index.text)

    # 2. Preset with override
    console.print("[bold cyan]2. Fetching with 'index' preset + link override...[/]")
    resp_override = requests.get(
        f"{BASE_API}/{TARGET_URL}",
        headers={
            "x-preset": "index",
            "x-retain-links": "all",
            "Accept": "application/json",
        },
        timeout=60,
    )
    out_override = OUTPUT_DIR / "preset_index_override.json"
    save_json(out_override, resp_override.text)

    console.print("\n[bold green]✔ Saved results:[/]")
    console.print(f"  📄 [link=file://{out_index}]{out_index.name}[/link]")
    console.print(f"  📄 [link=file://{out_override}]{out_override.name}[/link]")


if __name__ == "__main__":
    run_preset_demo()
