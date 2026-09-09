"""Example: Full pipeline with MarkdownChef, assertions, and artifact saving."""

import json
import shutil
from pathlib import Path

from jet.adapters.chonkie.markdown_chunker import (
    MarkdownChunkResult,
    chunk_markdown,
)
from rich.console import Console

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_MD = """
# Project Overview
This document describes the main features of our system.

## Installation
Install with:
```bash
pip install mypackage
```

## Features
| Feature       | Status      | Notes                  |
|---------------|-------------|------------------------|
| Fast search   | Done        | Uses vector index      |
| Batch upload  | In progress | Coming in v2.1         |
| Auth          | Done        | OAuth2 + API keys      |

## Code Example
Here is a simple Python helper:
```python
def process_data(items):
    results = []
    for item in items:
        if item.is_valid():
            results.append(item.transform())
    return results
```

## Conclusion
The system is ready for production use.
"""

EXPECTED_NUM_CHUNKS = 2
EXPECTED_NUM_TABLES = 1
EXPECTED_NUM_CODE_BLOCKS = 2
EXPECTED_NUM_IMAGES = 0
EXPECTED_CHUNK_1_TOKENS_MIN = 400
EXPECTED_CHUNK_1_TOKENS_MAX = 480
EXPECTED_CHUNK_2_TOKENS_MIN = 250
EXPECTED_CHUNK_2_TOKENS_MAX = 300
EXPECTED_PHRASES_CHUNK_1 = [
    "# Project Overview",
    "## Installation",
    "pip install mypackage",
    "## Features",
    "Fast search",
]
EXPECTED_PHRASES_CHUNK_2 = [
    "## Code Example",
    "def process_data",
    "## Conclusion",
]


def _save_results(result: MarkdownChunkResult, output_dir: Path) -> None:
    """Persist all artifacts including chunks, tables, code blocks, and summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save individual chunk files and JSON
    chunks_data = []
    for i, chunk in enumerate(result.chunks, 1):
        (output_dir / f"chunk_{i:02d}.txt").write_text(chunk.text, encoding="utf-8")
        chunks_data.append(
            {
                "index": i,
                "text": chunk.text,
                "token_count": chunk.token_count,
                "start_index": chunk.start_index,
                "end_index": chunk.end_index,
            }
        )
    (output_dir / "chunks.json").write_text(
        json.dumps(chunks_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Save extracted tables
    for i, table in enumerate(result.tables, 1):
        (output_dir / f"table_{i:02d}.md").write_text(table.content, encoding="utf-8")

    # Save extracted code blocks
    for i, code in enumerate(result.code_blocks, 1):
        lang = code.language or "txt"
        (output_dir / f"code_{i:02d}.{lang}").write_text(code.content, encoding="utf-8")

    # Save full content and summary
    (output_dir / "full_content.md").write_text(result.full_content, encoding="utf-8")
    summary = {
        "num_chunks": len(result.chunks),
        "num_tables": len(result.tables),
        "num_code_blocks": len(result.code_blocks),
        "num_images": len(result.images),
        "chunk_token_counts": [c.token_count for c in result.chunks],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main() -> None:
    console.rule("[bold]Example 03 – MarkdownChef + Full Content Pipeline[/bold]")

    result = chunk_markdown(SAMPLE_MD, chunk_size=512)

    # Assertions
    assert len(result.chunks) == EXPECTED_NUM_CHUNKS
    assert len(result.tables) == EXPECTED_NUM_TABLES
    assert len(result.code_blocks) == EXPECTED_NUM_CODE_BLOCKS
    assert len(result.images) == EXPECTED_NUM_IMAGES

    c1, c2 = result.chunks[0], result.chunks[1]
    assert EXPECTED_CHUNK_1_TOKENS_MIN <= c1.token_count <= EXPECTED_CHUNK_1_TOKENS_MAX
    assert EXPECTED_CHUNK_2_TOKENS_MIN <= c2.token_count <= EXPECTED_CHUNK_2_TOKENS_MAX

    for phrase in EXPECTED_PHRASES_CHUNK_1:
        assert phrase in c1.text
    for phrase in EXPECTED_PHRASES_CHUNK_2:
        assert phrase in c2.text

    for chunk in result.chunks:
        assert chunk.text.strip(), "Empty chunk found after filtering"

    console.print("[bold green]All assertions passed![/bold green]")

    _save_results(result, OUTPUT_DIR)

    console.print(f"\n[bold cyan]Generated Files:[/bold cyan]")
    for f in sorted(OUTPUT_DIR.glob("*")):
        console.print(f"  • [link={f.as_uri()}]{f.name}[/link]")


if __name__ == "__main__":
    main()
