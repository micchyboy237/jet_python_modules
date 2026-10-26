"""
HTML RAG Extractor using Trafilatura.

Extracts main content from HTML, cleans it, and chunks it for RAG purposes.
"""

import hashlib
import logging
from typing import Any, Dict, List, Optional

from jet.adapters.llama_cpp.chunking_utils import chunk_texts_with_data
from jet.adapters.llama_cpp.config import EMBED_MODEL
from jet.logger import logger
from trafilatura import extract, extract_with_metadata
from trafilatura.metadata import Document

LOGGER = logging.getLogger(__name__)


class HtmlRagChunk:
    """Represents a single chunk of extracted content with metadata."""

    def __init__(
        self,
        content: str,
        doc_id: str,
        chunk_index: int,
        start_idx: int,
        end_idx: int,
        num_tokens: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.content = content
        self.doc_id = doc_id
        self.chunk_index = chunk_index
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.num_tokens = num_tokens
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "num_tokens": self.num_tokens,
            "metadata": self.metadata,
        }


class HtmlRagExtractor:
    """
    Extracts clean text from HTML using Trafilatura and chunks it for RAG.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        embed_model: str = EMBED_MODEL,
        min_chunk_size: int = 32,
        favor_precision: bool = False,
        include_comments: bool = False,
        include_tables: bool = True,
        include_links: bool = False,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embed_model = embed_model
        self.min_chunk_size = min_chunk_size
        self.favor_precision = favor_precision
        self.include_comments = include_comments
        self.include_tables = include_tables
        self.include_links = include_links

    def extract_and_chunk(
        self,
        html_content: str,
        url: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> List[HtmlRagChunk]:
        if not html_content or not html_content.strip():
            logger.warning("Empty HTML content provided.")
            return []

        try:
            cleaned_text = extract(
                filecontent=html_content,
                url=url,
                favor_precision=self.favor_precision,
                include_comments=self.include_comments,
                include_tables=self.include_tables,
                include_links=self.include_links,
                output_format="txt",
                no_fallback=False,
            )
        except Exception as e:
            logger.error(f"Trafilatura extraction failed: {e}")
            return []

        if not cleaned_text:
            logger.warning("Trafilatura returned empty content.")
            return []

        if doc_id is None:
            source = url or cleaned_text[:100]
            doc_id = hashlib.md5(source.encode("utf-8")).hexdigest()

        try:
            chunk_results = chunk_texts_with_data(
                texts=cleaned_text,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                model=self.embed_model,
                ids=[doc_id],
                min_chunk_size=self.min_chunk_size,
                show_progress=False,
            )
        except Exception as e:
            logger.error(f"Chunking failed: {e}")
            return []

        rag_chunks = []
        for chunk_res in chunk_results:
            chunk_obj = HtmlRagChunk(
                content=chunk_res["content"],
                doc_id=chunk_res["doc_id"],
                chunk_index=chunk_res["chunk_index"],
                start_idx=chunk_res["start_idx"],
                end_idx=chunk_res["end_idx"],
                num_tokens=chunk_res["num_tokens"],
                metadata={
                    "source_url": url,
                    "extraction_method": "trafilatura",
                },
            )
            rag_chunks.append(chunk_obj)

        logger.info(f"Extracted {len(rag_chunks)} chunks from HTML.")
        return rag_chunks

    def extract_metadata(
        self,
        html_content: str,
        url: Optional[str] = None,
    ) -> Optional[Document]:
        try:
            doc = extract_with_metadata(
                filecontent=html_content,
                url=url,
                favor_precision=self.favor_precision,
                include_comments=self.include_comments,
                include_tables=self.include_tables,
                include_links=self.include_links,
                output_format="txt",
            )
            return doc
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            return None
