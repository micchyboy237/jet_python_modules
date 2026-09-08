"""Basic usage of HTMLAwareChunker on pre-cleaned Markdown content."""

import json
import shutil
from pathlib import Path

from jet.adapters.chonkie.html_chunkers import HTMLAwareChunker
from rich.console import Console

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Run chunker ---
chunker = HTMLAwareChunker(chunk_size=512)

markdown = """
# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems
to learn and improve from experience without being explicitly programmed.

## Types of ML

There are three main types: supervised, unsupervised, and reinforcement learning.

| Algorithm | Type | Use Case |
|-----------|------|----------|
| Linear Regression | Supervised | Prediction |
| K-Means | Unsupervised | Clustering |
| Q-Learning | RL | Game AI |

```python
def train_model(data):
    model = LinearRegression()
    model.fit(data.X, data.y)
    return model
```

Reinforcement learning has shown remarkable results in robotics and game playing.
"""

chunks = chunker.chunk(markdown)

# --- Save output ---
output_file = OUTPUT_DIR / "chunks.json"
serializable = [
    {
        "text": c.text,
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
