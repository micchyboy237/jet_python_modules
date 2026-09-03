from typing import Optional, TypedDict


class ChunkResult(TypedDict):
    """Core information for an individual chunk.

    Attributes:
        id: Unique chunk identifier.
        doc_id: Document ID (same as in meta).
        doc_index: Document index (same as in meta).
        chunk_index: Chunk order within the document.
        num_tokens: Number of tokens in this chunk.
        content: Text content of the chunk.
        start_idx: Start offset in source content.
        end_idx: End offset in source content.
        line_idx: Line number in the source.
        overlap_start_idx: Start index of overlap with previous chunk, if any.
        overlap_end_idx: End index of overlap with next chunk, if any.
    """

    id: str
    doc_id: str
    doc_index: int
    chunk_index: int
    num_tokens: int
    content: str
    start_idx: int
    end_idx: int
    line_idx: int
    overlap_start_idx: int | None
    overlap_end_idx: int | None


class MarkdownChunkMetadata(TypedDict):
    """Source-text index ranges for a markdown hierarchy chunk.

    Attributes:
        start_idx: Start offset including the header line.
        end_idx: End offset including the header line.
        body_start_idx: Start offset of body content only (excludes header).
        body_end_idx: End offset of body content only (excludes header).
    """

    start_idx: int
    end_idx: int
    body_start_idx: int
    body_end_idx: int


class MarkdownChunkResult(TypedDict):
    """Chunk result for markdown hierarchy-aware chunking.

    Attributes:
        id: Unique chunk identifier.
        doc_id: Document ID.
        doc_index: Document index in the input list.
        parent_id: ID of the parent header chunk, or None for top-level.
        header_doc_id: Unique ID for the header section this chunk belongs to.
        section_index: Sequential section counter within the document.
        chunk_index: Chunk order within the same header section.
        num_tokens: Token count including header tokens.
        header: Full header line (e.g., '## Setup').
        parent_header: Parent header line, or None for top-level sections.
        content: Body text content of the chunk (excludes header).
        level: Header nesting level (1-6).
        parent_level: Parent header nesting level, or None.
        metadata: Source-text index ranges for reconstruction and highlighting.
    """

    id: str
    doc_id: str
    doc_index: int
    parent_id: Optional[str]
    header_doc_id: str
    section_index: int
    chunk_index: int
    num_tokens: int
    header: str
    parent_header: Optional[str]
    content: str
    level: int
    parent_level: Optional[int]
    metadata: MarkdownChunkMetadata
