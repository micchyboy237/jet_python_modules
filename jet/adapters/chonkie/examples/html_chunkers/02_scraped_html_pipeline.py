"""End-to-end pipeline: raw HTML → Unstructured → hierarchy → smart chunking."""

import json
import shutil
from pathlib import Path

from jet.adapters.chonkie.html_chunkers import ScrapedHTMLPipeline
from rich.console import Console

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Run pipeline ---
pipeline = ScrapedHTMLPipeline(chunk_size=512)

raw_html = """
<html>
<body>
<nav><a href="/">Home</a></nav>
<h1>Climate Change Report 2025</h1>
<p>Global temperatures rose by 1.2°C above pre-industrial levels.</p>
<h2>Regional Impact</h2>
<p>The Arctic region is warming at nearly four times the global average rate.</p>
<table>
<tr><th>Region</th><th>Avg Temp Rise</th></tr>
<tr><td>Arctic</td><td>3.1°C</td></tr>
<tr><td>Tropics</td><td>0.8°C</td></tr>
</table>
<h2>Data Analysis</h2>
<pre><code class="python">
import pandas as pd
df = pd.read_csv("climate_data.csv")
print(df.describe())
</code></pre>
<p>Immediate action is required to limit warming to 1.5°C.</p>
<footer>© 2025 Climate Institute</footer>
</body>
</html>
"""

results = pipeline.process(raw_html, source_url="https://example.com/climate")

# --- Save output ---
output_file = OUTPUT_DIR / "pipeline_results.json"
serializable = [
    {
        "element_category": r.element_category,
        "breadcrumb": r.breadcrumb,
        "source_url": r.source_url,
        "page_number": r.page_number,
        "text": r.chunk.text,
        "token_count": r.chunk.token_count,
    }
    for r in results
]
output_file.write_text(json.dumps(serializable, indent=2))

# --- Display resource links ---
console.print(f"\n[bold]Saved files:[/]")
for f in sorted(OUTPUT_DIR.iterdir()):
    console.print(f"  📄 [link=file://{f}]{f.name}[/link]")
