"""Example: CodeChunker with large chunk size (2048 tokens)."""

import json
from pathlib import Path

from chonkie import CodeChunker
from jet.adapters.chonkie.markdown_chunker import (
    MarkdownChunkResult,
    remove_empty_chunks,
)
from rich.console import Console

console = Console()

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


def _save_results(result: MarkdownChunkResult, output_dir: Path) -> None:
    """Persist chunks and metadata to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_data = [
        {
            "index": i,
            "text": c.text,
            "token_count": c.token_count,
            "start_index": c.start_index,
            "end_index": c.end_index,
        }
        for i, c in enumerate(result.chunks, 1)
    ]
    (output_dir / "chunks.json").write_text(
        json.dumps(chunks_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "full_content.md").write_text(result.full_content, encoding="utf-8")
    console.print(f"[green]Saved to {output_dir}[/green]")


def main() -> None:
    console.rule("[bold]Example 01 – CodeChunker (chunk_size=2048)[/bold]")

    chunker = CodeChunker(
        language="markdown",
        tokenizer="character",
        chunk_size=2048,
        include_nodes=False,
    )
    chunks = remove_empty_chunks(chunker.chunk(SAMPLE_MD))
    result = MarkdownChunkResult(chunks=chunks, full_content=SAMPLE_MD)

    output_dir = Path(__file__).parent.parent / "generated" / "01_code_chunker_large"
    _save_results(result, output_dir)

    console.print(f"Generated {len(chunks)} chunks")


if __name__ == "__main__":
    main()
