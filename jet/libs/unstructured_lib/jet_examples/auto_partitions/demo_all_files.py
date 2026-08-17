"""Demonstration of auto-partitioning across multiple file types.

This module provides individual functions for partitioning different file types
found in the unstructured example-docs directory. Each function demonstrates
loading a specific file type, partitioning it, and saving the resulting
elements to a generated output directory.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from unstructured.partition.auto import partition

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Setup output directory as requested
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Base path for example documents relative to this project structure
# Adjust this path if your environment mounts example-docs differently
EXAMPLE_DOCS_DIR = (
    Path(__file__).resolve().parents[4] / "repo-libs" / "unstructured" / "example-docs"
)


def _save_elements(elements: list[Any], filename: str) -> None:
    """Serialize elements to JSON and save to OUTPUT_DIR.

    Args:
        elements: List of Element objects returned by partition.
        filename: Original filename used to name the output JSON.
    """
    output_path = OUTPUT_DIR / f"{filename}.json"
    try:
        # Convert elements to dict representation
        element_dicts = [e.to_dict() for e in elements]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(element_dicts, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(elements)} elements to {output_path.name}")
    except Exception as e:
        logger.error(f"Failed to save elements for {filename}: {e}")


def _run_partition(filename: str, **kwargs: Any) -> None:
    """Generic helper to partition a file and save results.

    Args:
        filename: Name of the file in EXAMPLE_DOCS_DIR.
        **kwargs: Additional arguments passed to partition().
    """
    file_path = EXAMPLE_DOCS_DIR / filename

    if not file_path.exists():
        logger.warning(f"File not found, skipping: {file_path}")
        return

    logger.info(f"Partitioning: {filename}")
    try:
        # Default to fast strategy for demo speed; auto will handle fallbacks
        kwargs.setdefault("strategy", "fast")
        # Disable table structure inference by default for speed in demo
        kwargs.setdefault("skip_infer_table_types", ["pdf", "jpg", "png", "heic"])

        elements = partition(filename=str(file_path), **kwargs)
        _save_elements(elements, filename)

    except Exception as e:
        logger.error(f"Error partitioning {filename}: {type(e).__name__}: {e}")


# --- Individual Example Functions ---


def demo_pdf() -> None:
    """Partition a PDF document."""
    _run_partition("fake-memo.pdf")


def demo_docx() -> None:
    """Partition a Word document."""
    _run_partition("fake-doc.docx")


def demo_pptx() -> None:
    """Partition a PowerPoint presentation."""
    _run_partition("fake-power-point.pptx")


def demo_xlsx() -> None:
    """Partition an Excel spreadsheet."""
    _run_partition("stanley-cups.xlsx")


def demo_html() -> None:
    """Partition an HTML file."""
    _run_partition("fake-html.html")


def demo_eml() -> None:
    """Partition an email file."""
    _run_partition("fake-email.eml")


def demo_image() -> None:
    """Partition an image file."""
    # Images often require hi_res or OCR; allowing auto to decide
    _run_partition("DA-1p.jpg", strategy="auto")


def demo_markdown() -> None:
    """Partition a Markdown file."""
    _run_partition("README.md")


def demo_json() -> None:
    """Partition a JSON file."""
    _run_partition("simple.json")


def demo_csv() -> None:
    """Partition a CSV file."""
    _run_partition("stanley-cups.csv")


def demo_txt() -> None:
    """Partition a plain text file."""
    _run_partition("fake-text.txt")


def demo_rst() -> None:
    """Partition a reStructuredText file."""
    _run_partition("README.rst")


def demo_org() -> None:
    """Partition an Org-mode file."""
    _run_partition("README.org")


def demo_epub() -> None:
    """Partition an EPUB ebook."""
    _run_partition("simple.epub")


def demo_odt() -> None:
    """Partition an OpenDocument text file."""
    _run_partition("fake.odt")


def demo_code() -> None:
    """Partition a source code file."""
    _run_partition("fake.go")


def demo_yaml() -> None:
    """Partition a YAML file."""
    _run_partition("simple.yaml")


def demo_xml() -> None:
    """Partition an XML file."""
    _run_partition("factbook.xml")


def run_all_demos() -> None:
    """Execute all available demo functions sequentially."""
    demos = [
        demo_pdf,
        demo_docx,
        demo_pptx,
        demo_xlsx,
        demo_html,
        demo_eml,
        demo_image,
        demo_markdown,
        demo_json,
        demo_csv,
        demo_txt,
        demo_rst,
        demo_org,
        demo_epub,
        demo_odt,
        demo_code,
        demo_yaml,
        demo_xml,
    ]

    logger.info(f"Starting auto-partition demos. Output: {OUTPUT_DIR}")
    logger.info(f"Example docs expected at: {EXAMPLE_DOCS_DIR}")

    for demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            logger.exception(f"Unhandled error in {demo_func.__name__}: {e}")

    logger.info("All demos completed.")


if __name__ == "__main__":
    run_all_demos()
