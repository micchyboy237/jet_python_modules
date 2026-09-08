"""Verify that element metadata (category, page, URL) survives chunking."""

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

# --- Run pipeline ---
pipeline = ScrapedHTMLPipeline(chunk_size=256, table_rows_per_chunk=2)

html_with_multiple_elements = """
<html>
<body>
<h1>Q3 Financial Results</h1>
<p>Revenue increased 15% compared to Q2.</p>
<table>
<tr><th>Metric</th><th>Q2</th><th>Q3</th></tr>
<tr><td>Revenue</td><td>$1.2M</td><td>$1.38M</td></tr>
<tr><td>Users</td><td>50K</td><td>62K</td></tr>
<tr><td>Churn</td><td>3.1%</td><td>2.8%</td></tr>
</table>
<p>Customer satisfaction scores reached an all-time high of 4.7/5.</p>
<pre><code class="sql">SELECT quarter, revenue FROM financials WHERE year = 2025;</code></pre>
</body>
</html>
"""

results = pipeline.process(
    html_with_multiple_elements, source_url="https://corp.example.com/q3"
)

# --- Save output ---
output_file = OUTPUT_DIR / "metadata_results.json"
serializable = [
    {
        "element_category": r.element_category,
        "source_url": r.source_url,
        "page_number": r.page_number,
        "text": r.chunk.text,
        "token_count": r.chunk.token_count,
        "start_index": r.chunk.start_index,
        "end_index": r.chunk.end_index,
    }
    for r in results
]
output_file.write_text(json.dumps(serializable, indent=2))

# --- Rich table preview ---
table = Table(title="Chunk Metadata Preview")
table.add_column("#", style="dim", width=3)
table.add_column("Category", style="cyan")
table.add_column("Tokens", justify="right")
table.add_column("Preview", max_width=50)

for i, item in enumerate(results):
    table.add_row(
        str(i),
        item.element_category,
        str(item.chunk.token_count),
        item.chunk.text[:50].strip() + "…",
    )
console.print(table)

# --- Display resource links ---
console.print(f"\n[bold]Saved files:[/]")
for f in sorted(OUTPUT_DIR.iterdir()):
    console.print(f"  📄 [link=file://{f}]{f.name}[/link]")
