"""Demo: Inject Page Script

Runs JavaScript in the page before extraction. Used for click-to-reveal
content like YouTube transcripts or 'Read More' buttons.
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

YOUTUBE_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
SCRIPT = "waitForSelector('ytd-video-description-transcript-section-renderer button').then((el) => el.click())"
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


def run_inject_demo():
    console.print("[bold cyan]Injecting script to reveal YouTube transcript...[/]")
    resp = requests.post(
        f"{BASE_API}/",
        data={"url": YOUTUBE_URL, "injectPageScript": SCRIPT},
        headers={"Accept": "application/json"},
        timeout=90,
    )
    out_file = OUTPUT_DIR / "transcript_extracted.json"
    save_json(out_file, resp.text)

    console.print("\n[bold green]✔ Saved results:[/]")
    console.print(f"  📄 [link=file://{out_file}]{out_file.name}[/link]")


if __name__ == "__main__":
    run_inject_demo()
