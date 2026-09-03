from typing import Literal, TypedDict


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
    """Extended metadata for markdown chunks with hierarchy indices."""

    start_idx: int
    end_idx: int
    body_start_idx: int
    body_end_idx: int
    line_idx: int


class MarkdownChunkResult(ChunkResult):
    """Chunk result enriched with markdown hierarchy information."""

    header: str
    parent_header: str | None
    level: int
    parent_level: int | None
    parent_id: str | None
    section_index: int
    metadata: MarkdownChunkMetadata


OverlapStrategy = Literal["none", "sentence", "token"]
