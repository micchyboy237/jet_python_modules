"""Smart chunker that adapts strategy based on document structure.

Delegates to existing proven strategies rather than reimplementing chunking logic.
Structure detection uses multiple signals with confidence scoring.
"""

import logging
from typing import List

from jet.adapters.llama_cpp.chunk_strategies.fixed_size_chunker import (
    TokenAwareFixedSizeChunker,
)
from jet.adapters.llama_cpp.chunk_strategies.model_utils import get_optimal_chunk_size
from jet.adapters.llama_cpp.chunk_strategies.sentence_chunker import (
    TokenAwareSentenceChunker,
)
from jet.adapters.llama_cpp.types import LLAMACPP_KEYS

logger = logging.getLogger(__name__)

_STRUCTURE_HEADERS_THRESHOLD = 3
_STRUCTURE_MIN_LINES = 15
_CODE_INDICATORS = ("```", "def ", "class ", "import ", "function ")
_TABLE_INDICATOR = "|"


class SmartChunker:
    """Structure-aware chunking that delegates to optimal sub-strategies.

    Detects document structure and routes to:
    - TokenAwareSentenceChunker for narrative prose
    - TokenAwareFixedSizeChunker for code/structured data
    - Hybrid approach for mixed documents

    Satisfies ChunkStrategy protocol for drop-in compatibility.
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
    ) -> List[str]:
        """Adaptively chunk text based on detected document structure.

        Args match ChunkStrategy protocol exactly for polymorphic compatibility.
        """
        if not text.strip():
            return []

        effective_chunk_size = chunk_size or self.default_chunk_size
        structure = self._detect_structure(text)

        logger.info(
            "SmartChunker: detected '%s' structure, chunk_size=%d, overlap=%d, "
            "min=%d, buffer=%d",
            structure,
            effective_chunk_size,
            chunk_overlap,
            min_chunk_size,
            buffer,
        )

        common_kwargs = dict(
            chunk_size=effective_chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
            buffer=buffer,
        )

        if structure == "code_heavy":
            logger.debug("Routing to TokenAwareFixedSizeChunker")
            return self._fixed_chunker.chunk(text=text, **common_kwargs)
        elif structure == "structured":
            logger.debug(
                "Routing structured doc to sentence chunker with reduced overlap"
            )
            # Structured docs benefit from less overlap to avoid repeating headers
            adjusted = {**common_kwargs, "chunk_overlap": max(0, chunk_overlap // 2)}
            return self._sentence_chunker.chunk(text=text, **adjusted)
        else:  # flat_narrative or mixed
            logger.debug("Routing to TokenAwareSentenceChunker")
            return self._sentence_chunker.chunk(text=text, **common_kwargs)

    def _detect_structure(self, text: str) -> str:
        """Detect document structure using multi-signal heuristic.

        Returns one of: 'code_heavy', 'structured', 'flat_narrative'.
        """
        lines = text.split("\n")
        non_empty_lines = [l for l in lines if l.strip()]
        total_lines = len(non_empty_lines)

        # Signal 1: Code density
        code_line_count = sum(
            1
            for l in non_empty_lines
            if any(indicator in l for indicator in _CODE_INDICATORS)
        )
        code_ratio = code_line_count / max(total_lines, 1)

        # Signal 2: Markdown headers
        header_count = sum(1 for l in non_empty_lines if l.strip().startswith("#"))

        # Signal 3: Table presence
        has_tables = any(_TABLE_INDICATOR in l for l in non_empty_lines)

        # Decision logic
        if code_ratio > 0.3:
            result = "code_heavy"
        elif (
            header_count >= _STRUCTURE_HEADERS_THRESHOLD
            and total_lines >= _STRUCTURE_MIN_LINES
        ):
            result = "structured"
        elif has_tables and code_ratio > 0.1:
            result = "code_heavy"  # Tables + code → treat as structured data
        else:
            result = "flat_narrative"

        logger.debug(
            "Structure detection: code_ratio=%.2f, headers=%d, lines=%d, "
            "has_tables=%s → %s",
            code_ratio,
            header_count,
            total_lines,
            has_tables,
            result,
        )
        return result
