"""Demo: Atomic-element-preserving chunking via SmartChunker.
Demonstrates how SmartChunker handles documents containing atomic elements
(tables, code snippets) that must NOT be split across chunk boundaries.
When atomic elements are detected, the chunker routes to FixedSizeChunker
to preserve their integrity, rather than sentence-splitting them.
Tests both element-detected and text-heuristic atomic detection.
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
MIXED_TEXT = """\
# Data Pipeline Overview
The ETL pipeline processes raw events through three stages.
| Stage | Input | Output | Latency |
|-------|-------|--------|---------|
| Extract | Raw logs | Normalized records | 50ms |
| Transform | Records | Enriched events | 120ms |
| Load | Events | Warehouse rows | 30ms |
The transform stage applies the following enrichment logic:
```python
def enrich_event(event: dict) -> dict:
    event["geo"] = lookup_geo(event["ip"])
    event["user_agent"] = parse_ua(event["ua_string"])
    event["session_id"] = compute_session(event["user_id"], event["ts"])
    return event
```
After enrichment, events are batch-loaded into the warehouse.
Batch size is configurable via the LOAD_BATCH_SIZE environment variable.
Default batch size is 1000 events with a 5-second flush interval.
"""
ELEMENTS_WITH_ATOMICS = [
    {"type": "Title", "text": "Data Pipeline Overview"},
    {
        "type": "NarrativeText",
        "text": "The ETL pipeline processes raw events through three stages.",
    },
    {
        "type": "Table",
        "text": "| Stage | Input | Output | Latency |\n|-------|-------|--------|---------|\n| Extract | Raw logs | Normalized records | 50ms |\n| Transform | Records | Enriched events | 120ms |\n| Load | Events | Warehouse rows | 30ms |",
    },
    {
        "type": "NarrativeText",
        "text": "The transform stage applies the following enrichment logic:",
    },
    {
        "type": "CodeSnippet",
        "text": 'def enrich_event(event: dict) -> dict:\n    event["geo"] = lookup_geo(event["ip"])\n    event["user_agent"] = parse_ua(event["ua_string"])\n    event["session_id"] = compute_session(event["user_id"], event["ts"])\n    return event',
    },
    {
        "type": "NarrativeText",
        "text": "After enrichment, events are batch-loaded into the warehouse.",
    },
    {
        "type": "NarrativeText",
        "text": "Batch size is configurable via the LOAD_BATCH_SIZE environment variable.",
    },
    {
        "type": "NarrativeText",
        "text": "Default batch size is 1000 events with a 5-second flush interval.",
    },
]
CHUNK_SIZE = 80
CHUNK_OVERLAP = 12
MIN_CHUNK_SIZE = 16
BUFFER = 4


def _check_atomic_integrity(chunks: list[str], label: str) -> list[str]:
    """Verify that table and code block markers are not split across chunks."""
    lines = [
        f"\n{'=' * 60}",
        f"🔍 Atomic Integrity Check: {label}",
        f"{'=' * 60}",
    ]
    print("\n".join(lines))

    table_complete = any("| Stage |" in c and "| Load |" in c for c in chunks)
    code_complete = any("def enrich_event" in c and "return event" in c for c in chunks)
    table_split = (
        sum(1 for c in chunks if "| Stage |" in c or "| Load |" in c) > 1
        and not table_complete
    )
    code_split = (
        sum(1 for c in chunks if "def enrich_event" in c or "return event" in c) > 1
        and not code_complete
    )

    table_msg = f"  Table intact in single chunk: {'✅' if table_complete else '❌'}"
    code_msg = f"  Code block intact in single chunk: {'✅' if code_complete else '❌'}"
    print(table_msg)
    print(code_msg)
    lines.extend([table_msg, code_msg])

    if table_split:
        split_msg = f"  ⚠️  Table appears split across chunks!"
        print(split_msg)
        lines.append(split_msg)
    if code_split:
        split_msg = f"  ⚠️  Code block appears split across chunks!"
        print(split_msg)
        lines.append(split_msg)
    return lines


def _print_chunks(label: str, chunks: list[str]) -> list[str]:
    """Print chunk details and return formatted lines."""
    lines = [
        f"\n{'=' * 60}",
        f"📄 {label}",
        f"{'=' * 60}",
        f"  Output: {len(chunks)} chunks",
    ]
    print("\n".join(lines))

    for i, chunk in enumerate(chunks):
        tok = estimate_tokens_safe(chunk, MODEL)
        preview = chunk[:80].replace("\n", "\\n")
        chunk_line = f"  [{i}] ({tok:>3d} tok) {preview}..."
        print(chunk_line)
        lines.append(chunk_line)
    return lines


def _get_element_route(elements: list[dict]) -> str:
    """Show what route SmartChunker will take for given elements."""
    types = [e.get("type", "") for e in elements]
    has_atomic = any(t in {"Table", "CodeSnippet", "Formula"} for t in types)
    has_sections = any(t in {"Title", "Header"} for t in types)
    if has_atomic:
        return "FixedSizeChunker (atomic_flat)"
    elif has_sections:
        return "SentenceChunker (structured, reduced overlap)"
    return "SentenceChunker (flat_narrative)"


def main() -> None:
    module_logger.info("=== Atomic-Element-Preserving Chunking Demo ===")
    print(
        f"Input: {len(MIXED_TEXT)} chars, ~{estimate_tokens_safe(MIXED_TEXT, MODEL)} tokens"
    )
    print(
        f"Config: chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}, min={MIN_CHUNK_SIZE}, buffer={BUFFER}"
    )

    # Path 1: Text Heuristic
    chunker_text = get_chunker("smart", model=MODEL)
    chunks_text = chunker_text.chunk(
        text=MIXED_TEXT,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        min_chunk_size=MIN_CHUNK_SIZE,
        buffer=BUFFER,
    )
    text_print_lines = _print_chunks(
        "Path 1: Text Heuristic (detects code fence + table)", chunks_text
    )
    text_integrity_lines = _check_atomic_integrity(chunks_text, "Text Heuristic")

    # Path 2: Element-Based
    chunker_elem = get_chunker("smart", model=MODEL)
    chunks_elem = chunker_elem.chunk(
        text=MIXED_TEXT,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        min_chunk_size=MIN_CHUNK_SIZE,
        buffer=BUFFER,
        elements=ELEMENTS_WITH_ATOMICS,
    )
    elem_print_lines = _print_chunks(
        "Path 2: Element-Based (explicit Table + CodeSnippet types)", chunks_elem
    )
    elem_integrity_lines = _check_atomic_integrity(chunks_elem, "Element-Based")

    # Summary
    print(f"\n{'=' * 60}")
    print("📊 SUMMARY")
    print(f"{'=' * 60}")
    text_table_ok = any("| Stage |" in c and "| Load |" in c for c in chunks_text)
    text_code_ok = any(
        "def enrich_event" in c and "return event" in c for c in chunks_text
    )
    elem_table_ok = any("| Stage |" in c and "| Load |" in c for c in chunks_elem)
    elem_code_ok = any(
        "def enrich_event" in c and "return event" in c for c in chunks_elem
    )

    summary_lines = [
        f"\n{'=' * 60}",
        "📊 SUMMARY",
        f"{'=' * 60}",
        f"  Text heuristic → FixedSizeChunker (code_heavy detection)",
        f"    Table intact: {'✅' if text_table_ok else '❌'}  Code intact: {'✅' if text_code_ok else '❌'}",
        f"  Element-based → {_get_element_route(ELEMENTS_WITH_ATOMICS)}",
        f"    Table intact: {'✅' if elem_table_ok else '❌'}  Code intact: {'✅' if elem_code_ok else '❌'}",
    ]
    for line in summary_lines[1:]:
        print(line)

    if not all([text_table_ok, text_code_ok, elem_table_ok, elem_code_ok]):
        warning = "  ⚠️  Neither chunker guarantees atomic element preservation."
        print(warning)
        summary_lines.append(warning)
    else:
        success = "  ✅ Both paths preserved atomic element integrity."
        print(success)
        summary_lines.append(success)

    print(f"\n{'=' * 60}")
    module_logger.info("Demo complete. Review atomic integrity checks above.")

    # Save all results
    all_lines = (
        [
            f"Input: {len(MIXED_TEXT)} chars, ~{estimate_tokens_safe(MIXED_TEXT, MODEL)} tokens",
            f"Config: chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}, min={MIN_CHUNK_SIZE}, buffer={BUFFER}",
            "",
        ]
        + text_print_lines
        + text_integrity_lines
        + elem_print_lines
        + elem_integrity_lines
        + summary_lines
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
