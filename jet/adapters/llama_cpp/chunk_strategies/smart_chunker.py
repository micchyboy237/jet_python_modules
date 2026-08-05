"""Smart chunker that adapts to document structure.

Supports two input modes:
1. Raw text: Uses heuristic structure detection (fast, no dependencies)
2. Unstructured elements: Uses semantic element types for accurate routing
"""

import logging
from typing import Any, Dict, List, Optional, Set

from jet.adapters.llama_cpp.chunk_strategies.fixed_size_chunker import (
    TokenAwareFixedSizeChunker,
)
from jet.adapters.llama_cpp.chunk_strategies.model_utils import get_optimal_chunk_size
from jet.adapters.llama_cpp.chunk_strategies.sentence_chunker import (
    TokenAwareSentenceChunker,
)
from jet.adapters.llama_cpp.types import LLAMACPP_KEYS

logger = logging.getLogger(__name__)

# Element type sets mirrored from document_parser for decoupled operation
_ATOMIC_TYPES: Set[str] = {"Table", "CodeSnippet", "Formula"}
_SECTION_TYPES: Set[str] = {"Title", "Header"}
_NARRATIVE_TYPES: Set[str] = {
    "NarrativeText",
    "Text",
    "UncategorizedText",
    "Paragraph",
    "ListItem",
    "BulletedText",
    "FigureCaption",
}

_STRUCTURE_HEADERS_THRESHOLD = 3
_STRUCTURE_MIN_LINES = 10
_CODE_INDICATORS = ("```", "def ", "class ", "import ", "function ")
_TABLE_INDICATOR = "|"


class SmartChunker:
    """Structure-aware chunking with optional unstructured element support.

    When elements are provided, uses semantic type classification for
    deterministic strategy selection. Falls back to text heuristics
    when only raw text is available.
    """

    def __init__(self, model: str | LLAMACPP_KEYS) -> None:
        self.model = model
        self.default_chunk_size = get_optimal_chunk_size(model)
        self._sentence_chunker = TokenAwareSentenceChunker(model)
        self._fixed_chunker = TokenAwareFixedSizeChunker(model)
        logger.info(
            "SmartChunker initialized for %s (default_chunk_size=%d)",
            model,
            self.default_chunk_size,
        )

    def chunk(
        self,
        text: str,
        chunk_size: int = 128,
        chunk_overlap: int = 0,
        min_chunk_size: int = 32,
        buffer: int = 0,
        elements: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """Adaptively chunk text based on document structure.

        Args:
            text: Raw text to chunk.
            chunk_size: Max tokens per chunk (0 or None → auto from model).
            chunk_overlap: Overlapping tokens between consecutive chunks.
            min_chunk_size: Minimum tokens for a chunk to be kept.
            buffer: Extra token margin reserved to avoid exceeding chunk_size.
            elements: Optional unstructured element dicts from partition().
                      When provided, enables semantic structure detection.

        Returns:
            List of chunk strings.
        """
        if not text.strip():
            return []

        effective_chunk_size = chunk_size or self.default_chunk_size

        common_kwargs = dict(
            chunk_size=effective_chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
            buffer=buffer,
        )

        # Route based on available structure signal
        if elements:
            structure = self._classify_from_elements(elements)
            logger.info(
                "SmartChunker: element-based structure='%s' (%d elements), "
                "chunk_size=%d",
                structure,
                len(elements),
                effective_chunk_size,
            )
            return self._chunk_by_structure(text, structure, common_kwargs)
        else:
            structure = self._detect_structure_from_text(text)
            logger.info(
                "SmartChunker: text-heuristic structure='%s', chunk_size=%d",
                structure,
                effective_chunk_size,
            )
            return self._chunk_by_structure(text, structure, common_kwargs)

    def _chunk_by_structure(
        self,
        text: str,
        structure: str,
        kwargs: dict,
    ) -> List[str]:
        """Dispatch to appropriate sub-strategy based on structure label."""
        if structure == "code_heavy":
            logger.debug("Routing to TokenAwareFixedSizeChunker")
            return self._fixed_chunker.chunk(text=text, **kwargs)
        elif structure == "structured":
            logger.debug("Routing to sentence chunker with reduced overlap")
            adjusted = {**kwargs, "chunk_overlap": max(0, kwargs["chunk_overlap"] // 2)}
            return self._sentence_chunker.chunk(text=text, **adjusted)
        elif structure == "atomic_flat":
            # Atomic elements (tables/code) mixed with narrative:
            # Use fixed-size to preserve atomic boundaries
            logger.debug("Routing atomic_flat to TokenAwareFixedSizeChunker")
            return self._fixed_chunker.chunk(text=text, **kwargs)
        else:  # flat_narrative, monolithic, mixed
            logger.debug("Routing to TokenAwareSentenceChunker")
            return self._sentence_chunker.chunk(text=text, **kwargs)

    # ── Element-based classification (deterministic) ──────────────────

    def _classify_from_elements(self, elements: List[Dict[str, Any]]) -> str:
        """Classify structure from unstructured element types.

        Mirrors classify_structure() from document_parser but returns
        labels compatible with _chunk_by_structure dispatch.
        """
        if not elements:
            return "flat_narrative"

        types = [e.get("type", "") for e in elements]
        has_sections = any(t in _SECTION_TYPES for t in types)
        has_atomic = any(t in _ATOMIC_TYPES for t in types)
        narrative_count = sum(1 for t in types if t in _NARRATIVE_TYPES)
        total = len(types)

        if has_sections and total > 3:
            return "structured"
        elif has_atomic and not has_sections:
            return "atomic_flat"
        elif narrative_count == total and total > 0:
            return "flat_narrative"
        elif total <= 2:
            return "flat_narrative"  # monolithic → treat as narrative
        else:
            return "structured"  # mixed → lean toward structured

    # ── Text-only heuristic classification (fallback) ─────────────────

    def _detect_structure_from_text(self, text: str) -> str:
        """Heuristic structure detection for raw text without elements."""
        lines = text.split("\n")
        non_empty_lines = [l for l in lines if l.strip()]
        total_lines = len(non_empty_lines)

        code_line_count = sum(
            1
            for l in non_empty_lines
            if any(indicator in l for indicator in _CODE_INDICATORS)
        )
        code_ratio = code_line_count / max(total_lines, 1)
        has_code_fence = any("```" in l for l in lines)

        header_count = sum(1 for l in non_empty_lines if l.strip().startswith("#"))
        has_tables = any(_TABLE_INDICATOR in l for l in non_empty_lines)

        is_code = (
            code_ratio > 0.15
            or has_code_fence
            or (code_line_count >= 3 and total_lines <= 50)
        )

        if is_code:
            result = "code_heavy"
        elif (
            header_count >= _STRUCTURE_HEADERS_THRESHOLD
            and total_lines >= _STRUCTURE_MIN_LINES
        ):
            result = "structured"
        elif has_tables and code_ratio > 0.1:
            result = "atomic_flat"
        else:
            result = "flat_narrative"

        logger.debug(
            "Text heuristic: code_ratio=%.2f, code_lines=%d/%d, fence=%s, "
            "headers=%d, tables=%s → %s",
            code_ratio,
            code_line_count,
            total_lines,
            has_code_fence,
            header_count,
            has_tables,
            result,
        )
        return result
