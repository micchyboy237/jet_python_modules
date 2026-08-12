"""Demo: Raw HTML Upload

Sends pre-existing HTML directly via the 'html' body field.
Reader skips the fetcher and runs the same conversion pipeline.
The optional 'url' field resolves relative links/images.
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


def save_json(path: Path, data: str | dict) -> None:
    """Parse and pretty-print JSON before saving for readability."""
    try:
        parsed = json.loads(data) if isinstance(data, str) else data
        path.write_text(
            json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except (json.JSONDecodeError, TypeError):
        path.write_text(data if isinstance(data, str) else str(data), encoding="utf-8")


def run_raw_html_demo():
    # Sample HTML with relative links/images to demonstrate url resolution
    sample_html = """<!DOCTYPE html>
<html lang="en">
<head><title>Test Article</title></head>
<body>
  <article>
    <h1>Raw HTML Upload Test</h1>
    <p>This content was sent directly as HTML, not fetched from a URL.</p>
    <img src="/images/hero.png" alt="Hero image with relative path">
    <a href="/docs/guide">Relative link to guide</a>
    <a href="https://jina.ai/reader">Absolute link</a>
  </article>
</body>
</html>"""

    payload = {
        "html": sample_html,
        "url": "https://jina.ai",  # Base URL for resolving relative paths
    }

    console.print(
        "[bold cyan]Uploading raw HTML with base URL for relative path resolution...[/]"
    )
    resp = requests.post(
        f"{BASE_API}/",
        json=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        timeout=60,
    )

    out_file = OUTPUT_DIR / "raw_html_converted.json"
    save_json(out_file, resp.text)

    console.print("\n[bold green]✔ Saved results:[/]")
    console.print(f"  📄 [link=file://{out_file}]{out_file.name}[/link]")


if __name__ == "__main__":
    run_raw_html_demo()
