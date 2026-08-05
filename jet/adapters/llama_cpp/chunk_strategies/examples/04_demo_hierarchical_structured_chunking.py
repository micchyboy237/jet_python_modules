"""Demo: Hierarchical structured chunking via SmartChunker with elements.
Demonstrates structure-aware chunking that respects document hierarchy:
sections → paragraphs → sentences. Uses unstructured element types for
deterministic routing when available, falls back to text heuristics otherwise.
Compares both paths on the same structured markdown document.
"""

import logging
import shutil
from pathlib import Path

from jet.adapters.llama_cpp.chunk_strategies import estimate_tokens_safe, get_chunker
from jet.adapters.llama_cpp.config import LLM_MODEL

# Rich console for styled resource links
from rich.console import Console

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG, format="%(name)s | %(levelname)s | %(message)s"
)
module_logger = logging.getLogger(__name__)

MODEL = LLM_MODEL
STRUCTURED_TEXT = """\
# Installation Guide
## Prerequisites
You need Python 3.10+ and CUDA 11.8 installed. Verify your setup before proceeding.
Ensure you have at least 8GB of VRAM available for model loading.
## Step 1: Clone the Repository
Use git to clone the project and install dependencies via pip.
Run the following commands in your terminal to get started.
## Step 2: Configure Environment Variables
Set the following variables in your .env file for database and API access.
The DATABASE_URL and API_KEY variables are required for all operations.
## Step 3: Run Migrations
Execute the migration script to initialize the database schema.
This creates all necessary tables and indexes for the application.
## Step 4: Start the Server
Launch the application with the production configuration flag enabled.
Monitor the logs directory for any startup errors or warnings.
## Troubleshooting
If you encounter CUDA out-of-memory errors, reduce the batch size in config.yaml.
Check the logs directory for detailed error traces and stack dumps.
For permission issues, verify that the data directory is writable by the current user.
"""
ELEMENTS = [
    {"type": "Title", "text": "Installation Guide"},
    {"type": "Header", "text": "Prerequisites"},
    {
        "type": "NarrativeText",
        "text": "You need Python 3.10+ and CUDA 11.8 installed. Verify your setup before proceeding.",
    },
    {
        "type": "NarrativeText",
        "text": "Ensure you have at least 8GB of VRAM available for model loading.",
    },
    {"type": "Header", "text": "Step 1: Clone the Repository"},
    {
        "type": "NarrativeText",
        "text": "Use git to clone the project and install dependencies via pip.",
    },
    {
        "type": "NarrativeText",
        "text": "Run the following commands in your terminal to get started.",
    },
    {"type": "Header", "text": "Step 2: Configure Environment Variables"},
    {
        "type": "NarrativeText",
        "text": "Set the following variables in your .env file for database and API access.",
    },
    {
        "type": "NarrativeText",
        "text": "The DATABASE_URL and API_KEY variables are required for all operations.",
    },
    {"type": "Header", "text": "Step 3: Run Migrations"},
    {
        "type": "NarrativeText",
        "text": "Execute the migration script to initialize the database schema.",
    },
    {
        "type": "NarrativeText",
        "text": "This creates all necessary tables and indexes for the application.",
    },
    {"type": "Header", "text": "Step 4: Start the Server"},
    {
        "type": "NarrativeText",
        "text": "Launch the application with the production configuration flag enabled.",
    },
    {
        "type": "NarrativeText",
        "text": "Monitor the logs directory for any startup errors or warnings.",
    },
    {"type": "Header", "text": "Troubleshooting"},
    {
        "type": "NarrativeText",
        "text": "If you encounter CUDA out-of-memory errors, reduce the batch size in config.yaml.",
    },
    {
        "type": "NarrativeText",
        "text": "Check the logs directory for detailed error traces and stack dumps.",
    },
    {
        "type": "NarrativeText",
        "text": "For permission issues, verify that the data directory is writable by the current user.",
    },
]
CHUNK_SIZE = 64
CHUNK_OVERLAP = 12
MIN_CHUNK_SIZE = 16
BUFFER = 4


def _print_chunks(label: str, chunks: list[str]) -> list[str]:
    """Print chunk details with token counts and return formatted lines."""
    lines = [
        f"\n{'=' * 60}",
        f"📄 {label}",
        f"{'=' * 60}",
        f"  Output: {len(chunks)} chunks",
    ]
    print("\n".join(lines[-4:]))

    total_tokens = 0
    for i, chunk in enumerate(chunks):
        tok = estimate_tokens_safe(chunk, MODEL)
        total_tokens += tok
        preview = chunk[:80].replace("\n", "\\n")
        chunk_line = f"  [{i}] ({tok:>3d} tok) {preview}..."
        print(chunk_line)
        lines.append(chunk_line)

    summary = f"  Total tokens across chunks: {total_tokens}"
    print(summary)
    lines.append(summary)
    return lines


def main() -> None:
    module_logger.info("=== Hierarchical Structured Chunking Demo ===")
    print(
        f"Input: {len(STRUCTURED_TEXT)} chars, ~{estimate_tokens_safe(STRUCTURED_TEXT, MODEL)} tokens"
    )
    print(
        f"Config: chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}, min={MIN_CHUNK_SIZE}, buffer={BUFFER}"
    )

    # Path 1: Text Heuristic
    chunker_text = get_chunker("smart", model=MODEL)
    chunks_text = chunker_text.chunk(
        text=STRUCTURED_TEXT,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        min_chunk_size=MIN_CHUNK_SIZE,
        buffer=BUFFER,
    )
    text_lines = _print_chunks("Path 1: Text Heuristic Detection", chunks_text)

    # Path 2: Element-Based
    chunker_elem = get_chunker("smart", model=MODEL)
    chunks_elem = chunker_elem.chunk(
        text=STRUCTURED_TEXT,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        min_chunk_size=MIN_CHUNK_SIZE,
        buffer=BUFFER,
        elements=ELEMENTS,
    )
    elem_lines = _print_chunks("Path 2: Element-Based Detection", chunks_elem)

    # Comparison
    print(f"\n{'=' * 60}")
    print("📊 COMPARISON")
    print(f"{'=' * 60}")
    text_tok = sum(estimate_tokens_safe(c, MODEL) for c in chunks_text)
    elem_tok = sum(estimate_tokens_safe(c, MODEL) for c in chunks_elem)

    comparison_lines = [
        f"\n{'=' * 60}",
        "📊 COMPARISON",
        f"{'=' * 60}",
        f"  Text heuristic:  {len(chunks_text)} chunks, {text_tok} total tokens",
        f"  Element-based:   {len(chunks_elem)} chunks, {elem_tok} total tokens",
        f"  Element path uses semantic types for deterministic routing",
        f"  Text path uses header/ratio heuristics as fallback",
    ]
    for line in comparison_lines[1:]:
        print(line)

    print(f"\n{'=' * 60}")
    module_logger.info(
        "Demo complete. Compare DEBUG logs for structure detection differences."
    )

    # Save all results
    all_lines = (
        [
            f"Input: {len(STRUCTURED_TEXT)} chars, ~{estimate_tokens_safe(STRUCTURED_TEXT, MODEL)} tokens",
            f"Config: chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}, min={MIN_CHUNK_SIZE}, buffer={BUFFER}",
            "",
        ]
        + text_lines
        + elem_lines
        + comparison_lines
    )
    summary_path = OUTPUT_DIR / "chunking_results.txt"
    summary_path.write_text("\n".join(all_lines), encoding="utf-8")
    console.print(
        f"💾 Results saved to [bold blue][link=file://{summary_path}]{summary_path.name}[/link][/bold blue]"
    )

    # Save individual chunk sets
    for label, chunks, dir_name in [
        ("text_heuristic", chunks_text, "chunks_text_heuristic"),
        ("element_based", chunks_elem, "chunks_element_based"),
    ]:
        chunks_dir = OUTPUT_DIR / dir_name
        chunks_dir.mkdir(parents=True, exist_ok=True)
        for i, chunk in enumerate(chunks):
            chunk_path = chunks_dir / f"chunk_{i:02d}.txt"
            tok = estimate_tokens_safe(chunk, MODEL)
            chunk_path.write_text(f"Tokens: {tok}\n\n{chunk}", encoding="utf-8")
        console.print(
            f"💾 {label} chunks saved to [bold blue][link=file://{chunks_dir}]{chunks_dir.name}/[/link][/bold blue] "
            f"({len(chunks)} files)"
        )


if __name__ == "__main__":
    main()
