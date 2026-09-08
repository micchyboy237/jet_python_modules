"""Async batch processing of multiple scraped HTML pages."""

import asyncio
import json
import shutil
from pathlib import Path

from jet.adapters.chonkie.html_chunkers import ScrapedHTMLPipeline
from rich.console import Console

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


async def main():
    pipeline = ScrapedHTMLPipeline(chunk_size=512)

    pages = [
        (
            "<html><body><h1>Renewable Energy</h1>"
            "<p>Solar capacity grew 30% year-over-year globally.</p>"
            "<p>Wind power now accounts for 10% of global electricity.</p></body></html>",
            "https://example.com/energy",
        ),
        (
            "<html><body><h1>Ocean Conservation</h1>"
            "<p>Marine protected areas have expanded to cover 8% of oceans.</p>"
            "<table><tr><th>Species</th><th>Status</th></tr>"
            "<tr><td>Blue Whale</td><td>Endangered</td></tr>"
            "<tr><td>Sea Turtle</td><td>Vulnerable</td></tr></table></body></html>",
            "https://example.com/ocean",
        ),
        (
            "<html><body>"
            "<p>This page has no headings at all. Just a wall of text about "
            "quantum computing applications in cryptography and drug discovery. "
            "The field continues to evolve rapidly with new breakthroughs.</p></body></html>",
            "https://example.com/quantum",
        ),
    ]

    tasks = [pipeline.aprocess(html, url) for html, url in pages]
    all_results = await asyncio.gather(*tasks)

    # --- Save per-page outputs ---
    for (_, url), results in zip(pages, all_results):
        safe_name = url.rstrip("/").split("/")[-1]
        output_file = OUTPUT_DIR / f"{safe_name}.json"
        serializable = [
            {
                "element_category": r.element_category,
                "text": r.chunk.text[:200],
                "token_count": r.chunk.token_count,
            }
            for r in results
        ]
        output_file.write_text(json.dumps(serializable, indent=2))
        categories = {}
        for item in results:
            categories[item.element_category] = (
                categories.get(item.element_category, 0) + 1
            )
        console.print(f"  [cyan]{url}[/]: {len(results)} chunks — {categories}")

    # --- Display resource links ---
    console.print(f"\n[bold]Saved files:[/]")
    for f in sorted(OUTPUT_DIR.iterdir()):
        console.print(f"  📄 [link=file://{f}]{f.name}[/link]")


if __name__ == "__main__":
    asyncio.run(main())
