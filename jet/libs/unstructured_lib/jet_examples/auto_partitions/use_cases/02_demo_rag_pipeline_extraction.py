"""
02_demo_rag_pipeline_extraction.py
Extracts clean text and tables from unknown sources optimized for vector embedding.
Preserves table HTML structure and parent-child element relationships.
"""

import json
import logging
from pathlib import Path

from unstructured.chunking.title import chunk_by_title
from unstructured.partition.auto import partition

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def extract_rag_ready_elements(filepath: Path) -> list[dict]:
    """Partition with hi_res for maximum fidelity, then chunk for embeddings."""
    logger.info(f"RAG extraction: {filepath.name}")

    elements = partition(
        filename=str(filepath),
        strategy="hi_res",
        skip_infer_table_types=[],  # Enable table extraction
        extract_image_block_types=["Image"],  # Capture images for multimodal RAG
        languages=["eng"],
    )

    # Chunk while preserving semantic boundaries
    chunks = chunk_by_title(
        elements, max_characters=1024, combine_text_under_n_chars=256
    )

    rag_records = []
    for chunk in chunks:
        record = {
            "chunk_id": chunk.id,
            "text": chunk.text,
            "metadata": {
                "source": filepath.name,
                "page": chunk.metadata.page_number,
                "filetype": chunk.metadata.filetype,
                "has_table": hasattr(chunk.metadata, "text_as_html")
                and bool(chunk.metadata.text_as_html),
                "table_html": getattr(chunk.metadata, "text_as_html", None),
                "parent_id": chunk.metadata.parent_id,
                "coordinates": chunk.metadata.coordinates.to_dict()
                if chunk.metadata.coordinates
                else None,
            },
        }
        rag_records.append(record)

    logger.info(f"Produced {len(rag_records)} RAG-ready chunks")
    return rag_records


def main():
    sample = Path(__file__).parent / "rag_sample.pdf"
    if not sample.exists():
        logger.warning("RAG sample not found. Creating placeholder.")
        sample.write_bytes(b"%PDF-1.4 placeholder")

    records = extract_rag_ready_elements(sample)

    output_path = Path(__file__).parent / "rag_output.jsonl"
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    logger.info(f"Saved RAG output to {output_path}")


if __name__ == "__main__":
    main()
