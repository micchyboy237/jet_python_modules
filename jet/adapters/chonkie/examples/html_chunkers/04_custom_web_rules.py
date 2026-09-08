"""Demonstrate custom RecursiveRules tuned for scraped web content."""

import json
import shutil
from pathlib import Path

from chonkie.types import RecursiveLevel, RecursiveRules
from jet.adapters.chonkie.html_chunkers import HTMLAwareChunker
from rich.console import Console

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Custom rules: prioritize H2/H3 splits, then paragraphs, then sentences
web_rules = RecursiveRules(
    rules=[
        RecursiveLevel(delimiters=["\n## ", "\n### "], include_delim="next"),
        RecursiveLevel(delimiters=["\n\n"], include_delim="next"),
        RecursiveLevel(delimiters=[". ", "! ", "? "], include_delim="prev"),
        RecursiveLevel(whitespace=True),
    ]
)

chunker = HTMLAwareChunker(chunk_size=256, min_chars_per_chunk=30)
# Override internal recursive chunker rules
chunker._recursive_chunker.rules = web_rules
chunker._recursive_chunker.min_characters_per_chunk = 30

messy_markdown = """
## Product Features

Our platform offers real-time analytics with sub-second latency.
It supports over 50 data connectors out of the box.

### Pricing Tiers

We offer three tiers: Free, Pro, and Enterprise.
Each tier includes unlimited users and API access.

#### Enterprise Add-ons

Custom SLAs, dedicated support, and on-premise deployment options
are available for Enterprise customers. Contact sales for details.
"""

chunks = chunker.chunk(messy_markdown)

# --- Save output ---
output_file = OUTPUT_DIR / "custom_rules_chunks.json"
serializable = [
    {
        "text": c.text.strip(),
        "token_count": c.token_count,
        "start_index": c.start_index,
        "end_index": c.end_index,
    }
    for c in chunks
]
output_file.write_text(json.dumps(serializable, indent=2))

# --- Display resource links ---
console.print(f"\n[bold]Saved files:[/]")
for f in sorted(OUTPUT_DIR.iterdir()):
    console.print(f"  📄 [link=file://{f}]{f.name}[/link]")
