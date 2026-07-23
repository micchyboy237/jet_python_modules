"""Header-aware markdown chunking.

Splits a markdown document along its headers first (so a chunk boundary never
lands mid-section), then greedily packs sections into token-budget-safe
chunks. Any single section that is itself larger than the budget is further
split by paragraph as a fallback.
"""

import logging
import re
from dataclasses import dataclass
from typing import Callable, List

logger = logging.getLogger("md_summarizer.chunking")

_HEADER_RE = re.compile(r"^(#{1,6})\s+.*$", re.MULTILINE)


@dataclass
class Chunk:
    text: str
    source_label: str  # file path, for logging/traceability


def _split_by_headers(text: str) -> List[str]:
    """Split on lines starting with '#', keeping each header attached to its body."""
    matches = list(_HEADER_RE.finditer(text))
    if not matches:
        return [text] if text.strip() else []

    sections = []
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(preamble)

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = text[start:end].strip()
        if section:
            sections.append(section)
    return sections


def _split_by_paragraph(text: str) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paras or [text]


def _pack(pieces: List[str], token_counter: Callable[[str], int], budget: int) -> List[str]:
    """Greedily concatenate consecutive pieces while staying under budget."""
    packed: List[str] = []
    buf = ""
    for piece in pieces:
        candidate = f"{buf}\n\n{piece}".strip() if buf else piece
        if token_counter(candidate) <= budget:
            buf = candidate
        else:
            if buf:
                packed.append(buf)
            buf = piece
    if buf:
        packed.append(buf)
    return packed


def chunk_markdown(
    text: str,
    token_counter: Callable[[str], int],
    token_budget: int,
    source_label: str = "",
) -> List[Chunk]:
    """Turn one markdown document into a list of budget-safe Chunks.

    Strategy: split by headers -> if a section alone exceeds budget, split
    that section by paragraph -> greedily pack the resulting safe pieces back
    together into as few budget-safe chunks as possible.
    """
    sections = _split_by_headers(text)
    if not sections:
        logger.warning("[%s] file is empty after stripping, skipping", source_label)
        return []

    safe_pieces: List[str] = []
    for section in sections:
        if token_counter(section) <= token_budget:
            safe_pieces.append(section)
            continue
        logger.info(
            "[%s] a section exceeds the %d-token budget alone; splitting it by paragraph",
            source_label, token_budget,
        )
        paras = _split_by_paragraph(section)
        safe_pieces.extend(_pack(paras, token_counter, token_budget))

    packed = _pack(safe_pieces, token_counter, token_budget)
    chunks = [Chunk(text=p, source_label=source_label) for p in packed]
    logger.info("[%s] chunked into %d piece(s)", source_label, len(chunks))
    return chunks
