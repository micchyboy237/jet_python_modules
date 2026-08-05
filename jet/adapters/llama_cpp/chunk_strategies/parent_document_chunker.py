"""Parent-Document Retrieval (PDR) chunking strategy.

Produces linked parent-child chunk pairs for hierarchical retrieval.
Children are embedded and indexed for precise search; parents are stored
separately and fetched at retrieval time to provide full context to the LLM.

This strategy delegates to SmartChunker for parent generation (leveraging
structure-aware chunking) and TokenAwareSentenceChunker for child generation.

Reuses existing jet.adapters.llama_cpp infrastructure:
    - get_optimal_chunk_size(): Auto-derives parent size from LLM context window
    - count_tokens(): Accurate token counting with local/server fallback
    - LLAMACPP_MODEL_EMBEDDING_SIZES: Caps child size to embed model context
    - EMBED_MODEL config: Identifies which embed model constrains child sizing
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

from jet.adapters.llama_cpp.chunk_strategies.model_utils import (
    get_optimal_chunk_size,
)
from jet.adapters.llama_cpp.chunk_strategies.sentence_chunker import (
    TokenAwareSentenceChunker,
)
from jet.adapters.llama_cpp.chunk_strategies.smart_chunker import SmartChunker
from jet.adapters.llama_cpp.config import EMBED_MODEL
from jet.adapters.llama_cpp.models import LLAMACPP_MODEL_EMBEDDING_SIZES
from jet.adapters.llama_cpp.token_utils import count_tokens
from jet.adapters.llama_cpp.types import LLAMACPP_KEYS

logger = logging.getLogger(__name__)

# Reserve tokens for special tokens / safety margin in embed model context
_EMBED_CTX_RESERVE = 16
_MIN_CHILD_SIZE_FLOOR = 64


class ParentDocumentChunker:
    """PDR chunker producing linked parent-child pairs.

    Does NOT implement the standard ChunkStrategy.chunk() signature because
    PDR inherently returns two collections, not one flat list. Instead,
    provides chunk_pdr() as the primary interface.

    Implements ChunkStrategy.chunk() as a compatibility shim that returns
    only child chunks (the indexable units). This allows use in pipelines
    that expect the standard protocol.

    Chunk sizes are auto-derived from model configuration when not explicitly
    provided:
        - Parent size: get_optimal_chunk_size(model) → ~8% of LLM context
        - Child size: parent_size // 8, capped to embed model context
    """

    def __init__(self, model: str | LLAMACPP_KEYS) -> None:
        self.model = model
        self._smart_chunker = SmartChunker(model)
        self._sentence_chunker = TokenAwareSentenceChunker(model)

        # Auto-derive sensible defaults from model context
        self._default_parent_size = get_optimal_chunk_size(model)
        self._default_child_size = max(
            _MIN_CHILD_SIZE_FLOOR, self._default_parent_size // 8
        )

        # Cap child size to embedding model's context window
        embed_ctx = LLAMACPP_MODEL_EMBEDDING_SIZES.get(EMBED_MODEL)
        if embed_ctx is not None:
            max_child = max(_MIN_CHILD_SIZE_FLOOR, embed_ctx - _EMBED_CTX_RESERVE)
            if self._default_child_size > max_child:
                logger.warning(
                    "Default child size %d exceeds embed model '%s' context %d; "
                    "capping to %d",
                    self._default_child_size,
                    EMBED_MODEL,
                    embed_ctx,
                    max_child,
                )
                self._default_child_size = max_child

        logger.info(
            "ParentDocumentChunker initialized for %s "
            "(default_parent=%d, default_child=%d, embed_model=%s)",
            model,
            self._default_parent_size,
            self._default_child_size,
            EMBED_MODEL,
        )

    # ── Primary PDR Interface ────────────────────────────────────────

    def chunk_pdr(
        self,
        text: str,
        parent_chunk_size: int | None = None,
        child_chunk_size: int | None = None,
        chunk_overlap: int = 0,
        min_child_size: int = 32,
        buffer: int = 0,
        elements: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Generate linked parent-child chunk pairs for PDR.

        Args:
            text: Raw document text.
            parent_chunk_size: Max tokens for parent chunks.
                None → auto-derive from LLM context via get_optimal_chunk_size().
            child_chunk_size: Max tokens for child chunks (indexed for search).
                None → auto-derive as parent_size // 8, capped by embed model context.
            chunk_overlap: Overlap between consecutive child chunks.
                Recommended: 0 (parent already provides boundary context).
            min_child_size: Minimum tokens for a child chunk to be kept.
            buffer: Token margin reserved to avoid exceeding child_chunk_size.
            elements: Optional unstructured element dicts. When provided,
                parents are generated via structure-aware SmartChunker.

        Returns:
            Dict with 'parents' and 'children' lists. Each child has
            'parent_id'; each parent has 'child_ids'.
        """
        if not text.strip():
            return {"parents": [], "children": []}

        # Resolve None → model-derived defaults
        if parent_chunk_size is None:
            parent_chunk_size = self._default_parent_size
        if child_chunk_size is None:
            child_chunk_size = self._default_child_size

        # Step 1: Generate parents using SmartChunker (structure-aware)
        parent_texts = self._smart_chunker.chunk(
            text=text,
            chunk_size=parent_chunk_size,
            chunk_overlap=0,  # Parents should NOT overlap
            min_chunk_size=min_child_size * 2,
            buffer=buffer,
            elements=elements,
        )

        parents: List[Dict[str, Any]] = []
        children: List[Dict[str, Any]] = []

        for p_idx, parent_text in enumerate(parent_texts):
            parent_id = f"par_{p_idx}_{uuid.uuid4().hex[:8]}"

            # Validate parent token count against budget
            actual_parent_tokens = count_tokens(parent_text, model=self.model)
            if actual_parent_tokens > parent_chunk_size:
                logger.warning(
                    "Parent %s exceeds budget: %d > %d tokens",
                    parent_id,
                    actual_parent_tokens,
                    parent_chunk_size,
                )

            # Step 2: Sub-chunk parent into children via sentence-aware chunker
            child_texts = self._sentence_chunker.chunk(
                text=parent_text,
                chunk_size=child_chunk_size,
                chunk_overlap=chunk_overlap,
                min_chunk_size=min_child_size,
                buffer=buffer,
            )

            child_ids_for_parent: List[str] = []
            for c_idx, child_text in enumerate(child_texts):
                child_id = f"ch_{p_idx}_{c_idx}_{uuid.uuid4().hex[:8]}"
                child_ids_for_parent.append(child_id)
                children.append(
                    {
                        "id": child_id,
                        "content": child_text,
                        "parent_id": parent_id,
                        "chunk_role": "child",
                        "parent_chunk_index": p_idx,
                        "child_index_within_parent": c_idx,
                    }
                )

            parents.append(
                {
                    "id": parent_id,
                    "content": parent_text,
                    "child_ids": child_ids_for_parent,
                    "chunk_role": "parent",
                    "parent_chunk_index": p_idx,
                    "num_tokens": actual_parent_tokens,
                }
            )

        logger.info(
            "PDR chunking complete: %d parents, %d children "
            "(parent_size=%d, child_size=%d, overlap=%d)",
            len(parents),
            len(children),
            parent_chunk_size,
            child_chunk_size,
            chunk_overlap,
        )
        return {"parents": parents, "children": children}

    # ── ChunkStrategy Compatibility Shim ─────────────────────────────

    def chunk(
        self,
        text: str,
        chunk_size: int = 256,
        chunk_overlap: int = 0,
        min_chunk_size: int = 32,
        buffer: int = 0,
    ) -> List[str]:
        """Compatibility shim: returns child chunk texts only.

        For full PDR functionality, use chunk_pdr() instead.
        This exists so ParentDocumentChunker satisfies ChunkStrategy protocol
        and can be used in generic pipelines that call .chunk().
        """
        result = self.chunk_pdr(
            text=text,
            parent_chunk_size=max(chunk_size * 8, self._default_parent_size),
            child_chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_child_size=min_chunk_size,
            buffer=buffer,
        )
        return [c["content"] for c in result["children"]]
