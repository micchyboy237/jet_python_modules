"""Demonstrate ScrapedHTMLPipeline with a list of URLs."""

import json
import shutil
from pathlib import Path

from jet.adapters.chonkie.html_chunkers import ScrapedHTMLPipeline
from rich.console import Console
from rich.table import Table

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pipeline = ScrapedHTMLPipeline(
    chunk_size=512,
    headless=True,
    use_cache=False,
    scroll_strategy="until_stable",
)

# Example: List of URLs
urls = [
    "https://deepeval.com/docs/metrics-introduction",
    "https://deepeval.com/docs/metrics-faithfulness",
]
console.print("\n[bold cyan]Processing List of URLs[/]")
multi_results = pipeline.process(urls)

# Save and display results
output_file = OUTPUT_DIR / "url_chunks.json"
serializable = [
    {
        "source_url": r.source_url,
        "element_category": r.element_category,
        "breadcrumb": r.breadcrumb,
        "text_preview": r.chunk.text[:150].replace("\n", " ") + "…",
        "token_count": r.chunk.token_count,
    }
    for r in multi_results
]
output_file.write_text(json.dumps(serializable, indent=2))

table = Table(title="URL Chunking Results")
table.add_column("Source", style="dim", max_width=30)
table.add_column("Category", style="cyan", max_width=15)
table.add_column("Tokens", justify="right", width=6)
table.add_column("Preview", max_width=50)

for item in serializable[:10]:  # Show first 10
    table.add_row(
        item["source_url"] or "(raw)",
        item["element_category"],
        str(item["token_count"]),
        item["text_preview"],
    )

console.print(table)
console.print(f"\n[bold]Saved files:[/]")
for f in sorted(OUTPUT_DIR.iterdir()):
    console.print(f"  📄 [link=file://{f}]{f.name}[/link]")
