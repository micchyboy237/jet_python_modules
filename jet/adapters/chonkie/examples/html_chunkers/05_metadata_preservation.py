"""Verify that hierarchy breadcrumbs and metadata survive chunking."""

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

# --- Run pipeline with deeply nested headers ---
pipeline = ScrapedHTMLPipeline(chunk_size=256, table_rows_per_chunk=2)

html_with_nested_headers = """
<html>
<body>
<h1>Q3 Financial Results</h1>
<p>Revenue increased 15% compared to Q2.</p>

<h2>Revenue Breakdown</h2>
<p>Enterprise revenue grew 22% while SMB remained flat.</p>

<h3>By Region</h3>
<table>
<tr><th>Region</th><th>Q2</th><th>Q3</th></tr>
<tr><td>North America</td><td>$800K</td><td>$950K</td></tr>
<tr><td>Europe</td><td>$300K</td><td>$350K</td></tr>
<tr><td>APAC</td><td>$100K</td><td>$80K</td></tr>
</table>

<h3>By Product Line</h3>
<p>SaaS subscriptions accounted for 70% of total revenue.</p>

<h2>Customer Metrics</h2>
<p>Customer satisfaction scores reached an all-time high of 4.7/5.</p>

<h3>Retention Analysis</h3>
<p>Net revenue retention improved to 118% from 112% in Q2.</p>

<pre><code class="sql">SELECT quarter, revenue FROM financials WHERE year = 2025;</code></pre>
</body>
</html>
"""

results = pipeline.process(
    html_with_nested_headers, source_url="https://corp.example.com/q3"
)

# --- Save output ---
output_file = OUTPUT_DIR / "metadata_results.json"
serializable = [
    {
        "element_category": r.element_category,
        "breadcrumb": r.breadcrumb,
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
table = Table(title="Chunk Metadata Preview (with Hierarchy)")
table.add_column("#", style="dim", width=3)
table.add_column("Category", style="cyan", max_width=16)
table.add_column("Breadcrumb", style="yellow", max_width=35)
table.add_column("Tokens", justify="right", width=6)
table.add_column("Preview", max_width=40)

for i, item in enumerate(results):
    table.add_row(
        str(i),
        item.element_category,
        item.breadcrumb or "(none)",
        str(item.chunk.token_count),
        item.chunk.text[:40].strip().replace("\n", " ") + "…",
    )
console.print(table)

# --- Display resource links ---
console.print(f"\n[bold]Saved files:[/]")
for f in sorted(OUTPUT_DIR.iterdir()):
    console.print(f"  📄 [link=file://{f}]{f.name}[/link]")
