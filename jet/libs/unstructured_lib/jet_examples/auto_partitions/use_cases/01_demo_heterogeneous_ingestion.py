"""
01_demo_heterogeneous_ingestion.py
Processes mixed datasets (PDF, DOCX, PNG) without custom routing logic.
Includes normalized output schema and per-file diagnostic reporting.
"""

import logging
from pathlib import Path
from typing import Any

from unstructured.documents.elements import Element
from unstructured.partition.auto import partition

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".png", ".jpg", ".txt", ".html"}


def normalize_element(el: Element, source_file: str) -> dict[str, Any]:
    """Flatten element into consistent schema regardless of source type."""
    return {
        "source_file": source_file,
        "element_id": el.id,
        "type": el.category,
        "text": el.text.strip(),
        "page_number": el.metadata.page_number,
        "filetype": el.metadata.filetype,
        "coordinates": el.metadata.coordinates.to_dict()
        if el.metadata.coordinates
        else None,
    }


def ingest_mixed_dataset(input_dir: Path) -> list[dict[str, Any]]:
    all_elements: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    files = sorted(
        f for f in input_dir.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    logger.info(f"Found {len(files)} heterogeneous files in {input_dir}")

    for filepath in files:
        try:
            logger.info(f"Processing: {filepath.name}")
            elements = partition(filename=str(filepath), strategy="auto")

            normalized = [normalize_element(el, filepath.name) for el in elements]
            all_elements.extend(normalized)

            diagnostics.append(
                {
                    "file": filepath.name,
                    "status": "success",
                    "element_count": len(elements),
                    "types": list(set(el.category for el in elements)),
                }
            )
        except Exception as e:
            logger.error(f"Failed to process {filepath.name}: {e}")
            diagnostics.append(
                {"file": filepath.name, "status": "error", "error": str(e)}
            )

    # Diagnostic summary
    success = sum(1 for d in diagnostics if d["status"] == "success")
    logger.info(f"Ingestion complete: {success}/{len(diagnostics)} files succeeded")
    for d in diagnostics:
        logger.info(
            f"  {d['file']}: {d['status']} ({d.get('element_count', 0)} elements)"
        )

    return all_elements


def main():
    dataset_dir = Path(__file__).parent / "mixed_dataset"
    dataset_dir.mkdir(exist_ok=True)

    # Create placeholder files if empty
    if not any(dataset_dir.iterdir()):
        logger.warning("No sample files found. Creating placeholders for demo.")
        (dataset_dir / "sample.pdf").write_bytes(b"%PDF-1.4 placeholder")
        (dataset_dir / "notes.txt").write_text("Sample text content for testing.")
        (dataset_dir / "scan.png").write_bytes(b"\x89PNG\r\n\x1a\n placeholder")

    results = ingest_mixed_dataset(dataset_dir)
    logger.info(f"Total normalized elements: {len(results)}")


if __name__ == "__main__":
    main()
