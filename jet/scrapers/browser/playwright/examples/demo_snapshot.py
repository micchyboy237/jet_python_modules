"""Demo: capture ARIA snapshots from a live URL and from a local HTML file."""

import asyncio
import shutil
from pathlib import Path

from jet.scrapers.browser.playwright import capture_snapshot
from rich.console import Console
from rich.text import Text

console = Console()

# Configuration
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_HTML = """
<!DOCTYPE html>
<html>
<head><title>Contact Page</title></head>
<body>
<nav>
    <a href="/home">Home</a>
    <a href="/about">About</a>
</nav>
<main>
    <h1>Contact Us</h1>
    <form>
        <label>Email <input type="email" name="email"></label>
        <button type="submit">Send</button>
    </form>
</main>
</body>
</html>
"""

saved_files = []


async def example_url() -> None:
    """Example 1: snapshot of a live URL (AI mode, limited depth)."""
    console.rule("[bold]Example 1: URL[/bold]")
    url = "https://playwright.dev/python/"

    snapshot = await capture_snapshot(
        url,
        mode="ai",
        depth=3,
        headless=False,
    )

    # Save output
    output_file = OUTPUT_DIR / "snapshot_playwright_url.txt"
    output_file.write_text(snapshot, encoding="utf-8")
    saved_files.append(output_file)

    console.print("[bold green]Snapshot (first 800 chars):[/bold green]")
    console.print(snapshot[:800])


async def example_html() -> None:
    """Example 2: snapshot of a local HTML file (default mode)."""
    console.rule("[bold]Example 2: HTML file[/bold]")

    # Save input
    input_file = OUTPUT_DIR / "contact.html"
    input_file.write_text(SAMPLE_HTML, encoding="utf-8")
    saved_files.append(input_file)
    console.print(f"[dim]Wrote sample HTML to: {input_file.name}[/dim]")

    snapshot = await capture_snapshot(
        input_file,
        mode="default",
        headless=True,
    )

    # Save output
    output_file = OUTPUT_DIR / "snapshot_contact_local.txt"
    output_file.write_text(snapshot, encoding="utf-8")
    saved_files.append(output_file)

    console.print("[bold green]Full snapshot:[/bold green]")
    console.print(snapshot)


async def main() -> None:
    await example_url()
    await example_html()

    console.rule("[bold]Generated Resources[/bold]")
    for f in saved_files:
        # Use Text.assemble to create a safe clickable link
        link_text = Text.assemble(
            ("📄 ", ""),
            (f.name, f"link file://{f}"),
        )
        console.print(link_text)


if __name__ == "__main__":
    asyncio.run(main())
