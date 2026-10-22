"""Demo: run JavaScript on a live URL and on a local HTML file."""

import asyncio
import json
import shutil
from pathlib import Path

from jet.scrapers.browser.playwright import evaluate
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
<head><title>Fruit Shop</title></head>
<body>
<h1>Fruit Shop</h1>
<ul id="fruits">
<li data-price="1.5">Apple</li>
<li data-price="0.8">Banana</li>
<li data-price="2.2">Cherry</li>
</ul>
</body>
</html>
"""

saved_files = []


async def example_url() -> None:
    """Example 1: evaluate JS on a live URL (no args, then with args)."""
    console.rule("[bold]Example 1: URL[/bold]")
    url = "https://quotes.toscrape.com/"

    title = await evaluate(url, "() => document.title", headless=True)
    console.print(f"[bold green]Page title:[/bold green] {title}")

    # Save output
    title_file = OUTPUT_DIR / "result_title.json"
    title_file.write_text(json.dumps({"title": title}), encoding="utf-8")
    saved_files.append(title_file)

    count = await evaluate(
        url,
        "(selector) => document.querySelectorAll(selector).length",
        args=[".quote"],
        headless=False,
    )
    console.print(f"[bold green]Number of quotes:[/bold green] {count}")

    # Save output
    count_file = OUTPUT_DIR / "result_quote_count.json"
    count_file.write_text(json.dumps({"count": count}), encoding="utf-8")
    saved_files.append(count_file)


async def example_html() -> None:
    """Example 2: evaluate JS on a local HTML file."""
    console.rule("[bold]Example 2: HTML file[/bold]")

    # Save input
    input_file = OUTPUT_DIR / "fruit_shop.html"
    input_file.write_text(SAMPLE_HTML, encoding="utf-8")
    saved_files.append(input_file)
    console.print(f"[dim]Wrote sample HTML to: {input_file.name}[/dim]")

    names = await evaluate(
        input_file,
        "() => [...document.querySelectorAll('#fruits li')].map(li => li.textContent)",
        headless=True,
    )
    console.print(f"[bold green]Fruit names:[/bold green] {names}")

    # Save output
    names_file = OUTPUT_DIR / "result_fruit_names.json"
    names_file.write_text(json.dumps(names), encoding="utf-8")
    saved_files.append(names_file)

    total = await evaluate(
        input_file,
        """(minPrice) => [...document.querySelectorAll('#fruits li')]
.map(li => parseFloat(li.dataset.price))
.filter(p => p >= minPrice)
.reduce((a, b) => a + b, 0)""",
        args=[1.0],
        headless=True,
    )
    console.print(f"[bold green]Total price (>= 1.0):[/bold green] {total}")

    # Save output
    total_file = OUTPUT_DIR / "result_total_price.json"
    total_file.write_text(json.dumps({"total": total}), encoding="utf-8")
    saved_files.append(total_file)


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
