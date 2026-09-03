"""
Demo: Markdown Hierarchy Chunking

Shows how to split markdown documents into semantically-aware chunks
that preserve header hierarchy, parent-child relationships, and source indices.
"""

import json
import logging

from jet.adapters.llama_cpp.chunking_utils import (
    chunk_markdown_hierarchy,
    chunk_markdown_hierarchy_with_data,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAMPLE_MARKDOWN = """\
# Jet Framework Guide

Welcome to the Jet framework documentation.

## Installation

Install via pip first. Then verify your installation.

### System Requirements

Python 3.10+ is required. GPU acceleration needs CUDA 11.8+.

### Optional Dependencies

For markdown processing, install extras. For RAG pipelines, add vector store deps.

## Configuration

Set environment variables before running. See the config reference for all options.

## API Reference

The main entry point is `jet.run()`. All adapters share a common interface.
"""


def demo_flat_chunks():
    """Simple usage: get list of chunk strings."""
    logger.info("=== Flat Chunk List ===")
    chunks = chunk_markdown_hierarchy(
        markdown_text=SAMPLE_MARKDOWN,
        chunk_size=64,
        min_chunk_size=8,
        show_progress=False,
    )
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i + 1} ({len(chunk)} chars) ---")
        print(chunk[:200])


def demo_rich_metadata():
    """Rich usage: get hierarchy metadata for RAG indexing."""
    logger.info("\n=== Rich Metadata Chunks ===")
    results = chunk_markdown_hierarchy_with_data(
        markdown_text=SAMPLE_MARKDOWN,
        ids=["jet-guide-v1"],
        chunk_size=64,
        min_chunk_size=8,
        show_progress=False,
    )
    logger.info(f"Generated {len(results)} chunks:")
    for res in results:
        summary = {
            "header": res["header"],
            "parent_header": res["parent_header"],
            "level": res["level"],
            "section_index": res["section_index"],
            "chunk_index": res["chunk_index"],
            "num_tokens": res["num_tokens"],
            "body_range": f"{res['metadata']['body_start_idx']}-{res['metadata']['body_end_idx']}",
            "line": res["metadata"]["line_idx"],
            "content_preview": res["content"][:80],
        }
        print(json.dumps(summary, indent=2))
        print("-" * 50)


def demo_multi_document():
    """Multiple documents with explicit IDs."""
    logger.info("\n=== Multi-Document Chunking ===")
    docs = [
        "# Doc A\nContent under A.\n## Sub A\nDeeper content.",
        "# Doc B\nDifferent document entirely.",
    ]
    results = chunk_markdown_hierarchy_with_data(
        markdown_text=docs,
        ids=["doc-a", "doc-b"],
        chunk_size=64,
        min_chunk_size=4,
        show_progress=False,
    )
    doc_groups: dict[str, int] = {}
    for r in results:
        doc_groups[r["doc_id"]] = doc_groups.get(r["doc_id"], 0) + 1
    for doc_id, count in doc_groups.items():
        logger.info(f"  {doc_id}: {count} chunks")


if __name__ == "__main__":
    demo_flat_chunks()
    demo_rich_metadata()
    demo_multi_document()
