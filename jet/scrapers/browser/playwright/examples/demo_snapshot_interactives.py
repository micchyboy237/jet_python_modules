"""Demo: capture ARIA snapshots filtered for interactive elements."""

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
<head><title>Mixed Content Page</title></head>
<body>
<header>
    <h1>Welcome to the Shop</h1>
    <nav>
        <a href="/home">Home</a>
        <a href="/products">Products</a>
        <a href="/contact">Contact</a>
    </nav>
</header>
<main>
    <section>
        <h2>Featured Items</h2>
        <p>Check out our latest arrivals.</p>
        <ul>
            <li>Item 1 <button>Add to Cart</button></li>
            <li>Item 2 <button>Add to Cart</button></li>
        </ul>
    </section>
    <form>
        <label>Newsletter Email: <input type="email" name="email"></label>
        <button type="submit">Subscribe</button>
    </form>
</main>
<footer>
    <p>&copy; 2026 Shop Inc.</p>
</footer>
</body>
</html>
"""

saved_files = []


async def main() -> None:
    console.rule("[bold]Interactive Elements Snapshot[/bold]")

    # Save input
    input_file = OUTPUT_DIR / "input_mixed.html"
    input_file.write_text(SAMPLE_HTML, encoding="utf-8")
    saved_files.append(input_file)
    console.print(f"[dim]Input saved to: {input_file.name}[/dim]")

    snapshot = await capture_snapshot(
        input_file, mode="ai", headless=True, filter_type="interactives"
    )

    # Save output
    output_file = OUTPUT_DIR / "snapshot_interactives.txt"
    output_file.write_text(snapshot, encoding="utf-8")
    saved_files.append(output_file)

    console.print("[bold green]Interactive Elements Snapshot:[/bold green]")
    console.print(snapshot)

    console.rule("[bold]Generated Resources[/bold]")
    for f in saved_files:
        link_text = Text.assemble(
            ("📄 ", ""),
            (f.name, f"link file://{f}"),
        )
        console.print(link_text)


if __name__ == "__main__":
    asyncio.run(main())
