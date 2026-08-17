"""
03_demo_legacy_digitization.py
Handles scanned PDFs with inconsistent text layers.
Implements OCR fallback chain, text density validation, and line merging.
"""

import logging
import re
from pathlib import Path

from unstructured.documents.elements import Text
from unstructured.partition.auto import partition

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

MIN_TEXT_DENSITY = 0.7  # Minimum ratio of non-whitespace chars to total chars


def compute_text_density(text: str) -> float:
    if not text:
        return 0.0
    stripped = re.sub(r"\s+", "", text)
    return len(stripped) / len(text)


def merge_split_lines(elements: list[Text]) -> list[Text]:
    """Merge lines that were incorrectly split by OCR (no period, short lines)."""
    merged: list[Text] = []
    buffer = ""

    for el in elements:
        text = el.text.strip()
        if not text:
            continue

        if buffer and (not buffer.endswith(".") or len(buffer) < 80):
            buffer += " " + text
        else:
            if buffer:
                merged.append(Text(text=buffer))
            buffer = text

    if buffer:
        merged.append(Text(text=buffer))

    logger.info(f"Merged {len(elements)} elements into {len(merged)} coherent blocks")
    return merged


def digitize_scanned_pdf(filepath: Path) -> list[Text]:
    # First pass: auto strategy
    logger.info(f"Digitizing scanned PDF: {filepath.name}")
    elements = partition(filename=str(filepath), strategy="auto", languages=["eng"])

    full_text = " ".join(el.text for el in elements if isinstance(el, Text))
    density = compute_text_density(full_text)
    logger.info(f"Initial text density: {density:.2f}")

    # If density too low, re-process with explicit OCR
    if density < MIN_TEXT_DENSITY:
        logger.warning("Low text density detected. Re-running with ocr_only strategy.")
        elements = partition(
            filename=str(filepath),
            strategy="ocr_only",
            languages=["eng"],
            ocr_languages="eng",
        )
        full_text = " ".join(el.text for el in elements if isinstance(el, Text))
        density = compute_text_density(full_text)
        logger.info(f"Post-OCR text density: {density:.2f}")

    text_elements = [el for el in elements if isinstance(el, Text)]
    return merge_split_lines(text_elements)


def main():
    scanned_pdf = Path(__file__).parent / "legacy_scan.pdf"
    if not scanned_pdf.exists():
        logger.warning("Legacy scan not found. Creating placeholder.")
        scanned_pdf.write_bytes(b"%PDF-1.4 placeholder")

    cleaned_blocks = digitize_scanned_pdf(scanned_pdf)
    for i, block in enumerate(cleaned_blocks[:3]):
        logger.info(f"Block [{i}]: {block.text[:120]}...")


if __name__ == "__main__":
    main()
