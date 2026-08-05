# jet_python_modules/jet/adapters/llama_cpp/chunk_strategies/smart_chunker.py
"""Smart chunker that adapts to document structure.

Supports two input modes:
1. Raw text: Uses heuristic structure detection + sub-chunker delegation
2. Unstructured elements: Delegates to document_parser.chunk_rag_context()
   for true atomic-element-preserving hierarchical chunking

Retrieval-type awareness (text-only path):
- dense: Default overlap (preserves boundary context for vector retrieval)
- sparse: Zero overlap (BM25/SPLADE don't benefit from duplicated tokens)
- hybrid: Zero overlap (reranker handles boundary continuity)
"""

import logging
from typing import Any, Dict, List, Literal, Optional, Set

from jet.adapters.llama_cpp.chunk_strategies.fixed_size_chunker import (
    TokenAwareFixedSizeChunker,
)
from jet.adapters.llama_cpp.chunk_strategies.model_utils import get_optimal_chunk_size
from jet.adapters.llama_cpp.chunk_strategies.sentence_chunker import (
    TokenAwareSentenceChunker,
)
from jet.adapters.llama_cpp.types import LLAMACPP_KEYS

logger = logging.getLogger(__name__)

# Element type sets for text-heuristic fallback only
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

RetrievalType = Literal["dense", "sparse", "hybrid"]


class SmartChunker:
    """Structure-aware chunking with optional unstructured element support.

    When elements are provided, delegates to document_parser.chunk_rag_context()
    which treats atomic elements (tables, code, formulas) as indivisible units.
    Falls back to text heuristics + sub-chunker delegation when only raw text
    is available.
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
        retrieval_type: RetrievalType = "dense",
    ) -> List[str]:
        """Adaptively chunk text based on document structure.

        Args:
            text: Raw text to chunk.
            chunk_size: Max tokens per chunk (0 or None → auto from model).
            chunk_overlap: Overlapping tokens between consecutive chunks.
                          Automatically set to 0 for sparse/hybrid retrieval.
            min_chunk_size: Minimum tokens for a chunk to be kept.
            buffer: Extra token margin reserved to avoid exceeding chunk_size.
            elements: Optional unstructured element dicts from partition().
                      When provided, delegates to chunk_rag_context() for
                      atomic-element-preserving hierarchical chunking.
            retrieval_type: Downstream retrieval method. Determines overlap:
                           - "dense": Use configured chunk_overlap (default)
                           - "sparse": Force overlap=0 (BM25/SPLADE)
                           - "hybrid": Force overlap=0 (reranker handles boundaries)

        Returns:
            List of chunk strings.
        """
        if not text.strip():
            return []

        effective_chunk_size = chunk_size or self.default_chunk_size

        # ── Element path: delegate to document_parser ─────────────────
        if elements:
            return self._chunk_with_elements(
                elements=elements,
                max_tokens=effective_chunk_size,
                overlap_tokens=chunk_overlap,
                retrieval_type=retrieval_type,
            )

        # ── Text-only path: heuristic + sub-chunker delegation ────────
        effective_overlap = chunk_overlap
        if retrieval_type in ("sparse", "hybrid"):
            if chunk_overlap > 0:
                logger.info(
                    "Retrieval type '%s': forcing overlap %d → 0",
                    retrieval_type,
                    chunk_overlap,
                )
            effective_overlap = 0

        common_kwargs = dict(
            chunk_size=effective_chunk_size,
            chunk_overlap=effective_overlap,
            min_chunk_size=min_chunk_size,
            buffer=buffer,
        )

        structure = self._detect_structure_from_text(text)
        logger.info(
            "SmartChunker: text-heuristic structure='%s', retrieval='%s', "
            "chunk_size=%d, overlap=%d",
            structure,
            retrieval_type,
            effective_chunk_size,
            effective_overlap,
        )
        return self._chunk_by_structure(text, structure, common_kwargs)

    # ── Element-based chunking via document_parser ────────────────────

    def _chunk_with_elements(
        self,
        elements: List[Dict[str, Any]],
        max_tokens: int,
        overlap_tokens: int,
        retrieval_type: RetrievalType,
    ) -> List[str]:
        """Delegate to document_parser.chunk_rag_context for atomic-safe chunking."""
        try:
            from jet.adapters.unstructured.document_parser import chunk_rag_context
        except ImportError as exc:
            logger.warning(
                "document_parser not available (%s); falling back to text heuristic",
                exc,
            )
            # Fallback: concatenate element texts and use text-heuristic path
            combined = "\n\n".join(
                e.get("text", "") for e in elements if e.get("text", "").strip()
            )
            structure = self._classify_from_elements(elements)
            kwargs = dict(
                chunk_size=max_tokens,
                chunk_overlap=overlap_tokens if retrieval_type == "dense" else 0,
                min_chunk_size=32,
                buffer=0,
            )
            return self._chunk_by_structure(combined, structure, kwargs)

        # Adjust overlap for retrieval type
        effective_overlap = overlap_tokens
        if retrieval_type in ("sparse", "hybrid"):
            effective_overlap = 0

        logger.info(
            "SmartChunker: delegating to chunk_rag_context (%d elements, "
            "max_tokens=%d, overlap=%d, retrieval='%s')",
            len(elements),
            max_tokens,
            effective_overlap,
            retrieval_type,
        )

        chunk_dicts = chunk_rag_context(
            elements=elements,
            max_tokens=max_tokens,
            overlap_tokens=effective_overlap,
            model=self.model,
        )

        chunks = [c["text"] for c in chunk_dicts if c.get("text", "").strip()]

        strategies_used = sorted({c.get("strategy", "unknown") for c in chunk_dicts})
        logger.info(
            "SmartChunker: chunk_rag_context produced %d chunks, strategies=%s",
            len(chunks),
            strategies_used,
        )
        return chunks

    # ── Text-only sub-chunker dispatch ────────────────────────────────

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
            logger.debug("Routing atomic_flat to TokenAwareFixedSizeChunker")
            return self._fixed_chunker.chunk(text=text, **kwargs)
        else:
            logger.debug("Routing to TokenAwareSentenceChunker")
            return self._sentence_chunker.chunk(text=text, **kwargs)

    # ── Element classification (fallback only) ────────────────────────

    def _classify_from_elements(self, elements: List[Dict[str, Any]]) -> str:
        """Classify structure from element types. Used only as fallback
        when document_parser is unavailable."""
        if not elements:
            return "flat_narrative"
        types = [e.get("type", "") for e in elements]
        has_sections = any(t in _SECTION_TYPES for t in types)
        has_atomic = any(t in _ATOMIC_TYPES for t in types)
        narrative_count = sum(1 for t in types if t in _NARRATIVE_TYPES)
        total = len(types)

        if has_atomic:
            return "atomic_flat"
        elif has_sections and total > 3:
            return "structured"
        elif narrative_count == total and total > 0:
            return "flat_narrative"
        elif total <= 2:
            return "flat_narrative"
        else:
            return "structured"

    # ── Text-only heuristic classification ────────────────────────────

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
