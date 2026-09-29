"""PDF extraction adapter using Docling for advanced structural understanding.

Docling provides a unified document representation that preserves tables,
code blocks, formulas, and reading order, which is critical for high-quality RAG.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from docling.document_converter import DocumentConverter
from docling_core.types.doc import DoclingDocument
from jet_telemetry import tool

logger = logging.getLogger(__name__)


class PdfExtractor:
    """Extracts structured content from PDF files using Docling."""

    def __init__(self) -> None:
        """Initialize the Docling converter."""
        self._converter: Optional[DocumentConverter] = None

    @property
    def converter(self) -> DocumentConverter:
        if self._converter is None:
            logger.info("Initializing Docling DocumentConverter...")
            self._converter = DocumentConverter()
        return self._converter

    @tool(
        name="extract-pdf-content",
        description="Extracts structured text and elements from a PDF file.",
    )
    def extract_from_path(self, pdf_path: str | Path) -> DoclingDocument:
        """Convert a PDF file into a structured DoclingDocument.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            A DoclingDocument object containing structured elements.
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        logger.info(f"Extracting content from: {path.name}")
        try:
            result = self.converter.convert(path)
            doc = result.document
            logger.info(
                f"Successfully extracted document '{doc.name}' with "
                f"{len(doc.texts)} text items, {len(doc.tables)} tables, "
                f"and {len(doc.pictures)} pictures."
            )
            return doc
        except Exception as e:
            logger.error(f"Failed to extract PDF content: {e}", exc_info=True)
            raise

    @tool(
        name="export-doc-to-markdown",
        description="Exports a DoclingDocument to Markdown format.",
    )
    def export_to_markdown(self, doc: DoclingDocument) -> str:
        """Export a DoclingDocument to Markdown format.

        This format is highly compatible with LangChain splitters and
        preserves structural cues like headers and lists.

        Args:
            doc: The DoclingDocument to export.

        Returns:
            A Markdown string representation of the document.
        """
        try:
            markdown_content = doc.export_to_markdown()
            logger.debug(
                f"Exported document to Markdown ({len(markdown_content)} chars)"
            )
            return markdown_content
        except Exception as e:
            logger.error(f"Failed to export to Markdown: {e}", exc_info=True)
            raise

    @tool(
        name="extract-doc-elements",
        description="Extracts structured elements for custom processing.",
    )
    def extract_elements(self, doc: DoclingDocument) -> List[Dict[str, Any]]:
        """Extract structured elements (text, tables, code) for custom processing.

        Args:
            doc: The DoclingDocument to process.

        Returns:
            A list of dictionaries representing document elements with their type and content.
        """
        elements = []

        # Iterate through the document body to maintain reading order
        for item, level in doc.iterate_items():
            elem_type = type(item).__name__
            content = ""
            metadata = {"level": level, "type": elem_type}

            if hasattr(item, "text"):
                content = item.text
            elif hasattr(item, "orig"):
                content = item.orig

            # Handle specific structural elements
            if elem_type == "TableItem":
                content = item.export_to_html(doc)
                metadata["subtype"] = "table"
            elif elem_type == "CodeItem":
                metadata["subtype"] = "code"
                if hasattr(item, "code_language"):
                    metadata["language"] = item.code_language
            elif elem_type == "FormulaItem":
                metadata["subtype"] = "formula"

            if content:
                elements.append({"content": content, "metadata": metadata})

        logger.info(f"Extracted {len(elements)} structured elements from document.")
        return elements
