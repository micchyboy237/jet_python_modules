"""Generic document parser with hierarchical RAG chunking for small-context LLMs."""

import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional, Set

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"

logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger("doc_parser")
logger.setLevel(logging.INFO)
_formatter = logging.Formatter("%(asctime)s [%(levelname)-8s] %(name)s | %(message)s")
_stream = logging.StreamHandler(sys.stdout)
_stream.setFormatter(_formatter)
_file = logging.FileHandler(
    "/Users/jethroestrada/Desktop/External_Projects/"
    "Jet_Projects/JetScripts/test/__sample.log",
    mode="w",
    encoding="utf-8",
)
_file.setFormatter(_formatter)
logger.addHandler(_stream)
logger.addHandler(_file)

for _noisy in ("pdfminer", "urllib3", "PIL", "fontTools", "opencv"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

try:
    from unstructured.partition.auto import partition

    logger.info("✅ unstructured.partition.auto imported successfully")
except ImportError as e:
    logger.critical(f"❌ Missing dependency: {e}")
    logger.critical("   Fix: pip install 'unstructured[all-docs]'")
    sys.exit(1)

# ✅ NEW: Import accurate token counting and model context utilities
try:
    from jet.adapters.llama_cpp.model_utils import get_model_ctx_embd_size
    from jet.adapters.llama_cpp.token_utils import count_tokens

    logger.info("✅ llama_cpp token_utils and model_utils imported successfully")
    _HAS_LLAMA_CPP_UTILS = True
except ImportError as e:
    logger.warning(
        f"⚠️ llama_cpp utils not available ({e}). "
        "Falling back to heuristic token estimation."
    )
    _HAS_LLAMA_CPP_UTILS = False

RAG_CONTEXT_TYPES: Set[str] = {
    "NarrativeText",
    "Text",
    "UncategorizedText",
    "Paragraph",
    "Title",
    "Header",
    "ListItem",
    "BulletedText",
    "Table",
    "FigureCaption",
    "Image",
    "CodeSnippet",
    "Formula",
    "EmailAddress",
    "Address",
    "FormKeysValues",
    "Form",
}
ATOMIC_TYPES: Set[str] = {"Table", "CodeSnippet", "Formula"}
SECTION_TYPES: Set[str] = {"Title", "Header"}


def estimate_tokens(text: str, model: Optional[str] = None) -> int:
    """
    Count tokens accurately using the LLM tokenizer when available,
    falling back to ~4 chars/token heuristic otherwise.

    Args:
        text: Text to count tokens for.
        model: Optional llama.cpp model key. Uses default LLM_MODEL if None.

    Returns:
        Token count (always >= 1 for non-empty text).
    """
    if not text:
        return 0
    if _HAS_LLAMA_CPP_UTILS:
        try:
            return count_tokens(text, add_special=False, model=model)
        except Exception as e:
            logger.debug(f"count_tokens failed, falling back to heuristic: {e}")
    return max(1, len(text) // 4)


def auto_chunk_size(model_key: Optional[str] = None) -> int:
    """
    Derive optimal chunk size from model context window.

    Uses ~10% of context as max chunk size, clamped to [128, 1024].
    Falls back to 400 if model info is unavailable.

    Args:
        model_key: llama.cpp model key (e.g., "qwen3.5:2b").
                   Uses default LLM_MODEL from config if None.

    Returns:
        Recommended max_tokens per chunk.
    """
    if not _HAS_LLAMA_CPP_UTILS:
        return 400
    try:
        info = get_model_ctx_embd_size(model_key)
        ctx = info.get("ctx", 0)
        if ctx <= 0:
            logger.warning(
                f"auto_chunk_size | model '{model_key}' reports ctx={ctx}, "
                "falling back to default 400"
            )
            return 400
        derived = min(max(ctx // 10, 128), 1024)
        logger.info(
            f"auto_chunk_size | model='{model_key}' ctx={ctx} → "
            f"recommended max_tokens={derived}"
        )
        return derived
    except ValueError as e:
        logger.warning(
            f"auto_chunk_size | could not resolve model '{model_key}': {e}. "
            "Falling back to default 400"
        )
        return 400


def classify_structure(elements: List[Dict[str, Any]]) -> str:
    """Classify document structure to select chunking strategy."""
    if not elements:
        return "empty"
    types = [e.get("type", "") for e in elements]
    has_sections = any(t in SECTION_TYPES for t in types)
    has_atomic = any(t in ATOMIC_TYPES for t in types)
    narrative_count = sum(1 for t in types if t == "NarrativeText")
    total = len(types)
    if has_sections and total > 3:
        return "structured"
    elif has_atomic and not has_sections:
        return "atomic_flat"
    elif narrative_count == total and total > 0:
        return "flat_narrative"
    elif total <= 2:
        return "monolithic"
    else:
        return "mixed"


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    """Split text into sentences. Returns single-item list if no boundaries found."""
    parts = _SENTENCE_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()] or [text.strip()]


def _merge_up_to_budget(
    texts: List[str],
    max_tokens: int,
    overlap_tokens: int,
    model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Merge sequential text fragments into chunks respecting token budget."""
    chunks: List[Dict[str, Any]] = []
    current_parts: List[str] = []
    current_tokens = 0

    for text in texts:
        text_tokens = estimate_tokens(text, model=model)

        if text_tokens > max_tokens:
            # Flush accumulated parts before handling oversized text
            if current_parts:
                chunks.append(
                    {
                        "text": "\n\n".join(current_parts),
                        "token_count": current_tokens,
                    }
                )
                current_parts = []
                current_tokens = 0

            # Split oversized text by words
            words = text.split()
            sub_parts: List[str] = []
            sub_tokens = 0

            for word in words:
                w_tok = estimate_tokens(word, model=model)
                if sub_tokens + w_tok > max_tokens and sub_parts:
                    chunks.append(
                        {
                            "text": " ".join(sub_parts),
                            "token_count": sub_tokens,
                        }
                    )
                    if overlap_tokens > 0:
                        overlap_text = " ".join(sub_parts[-3:])
                        sub_parts = [overlap_text]
                        sub_tokens = estimate_tokens(overlap_text, model=model)
                    else:
                        sub_parts = []
                        sub_tokens = 0
                sub_parts.append(word)
                sub_tokens += w_tok

            if sub_parts:
                chunks.append(
                    {
                        "text": " ".join(sub_parts),
                        "token_count": sub_tokens,
                    }
                )
            continue

        if current_tokens + text_tokens > max_tokens and current_parts:
            chunks.append(
                {
                    "text": "\n\n".join(current_parts),
                    "token_count": current_tokens,
                }
            )
            if overlap_tokens > 0:
                overlap_parts: List[str] = []
                overlap_tok = 0
                for part in reversed(current_parts):
                    pt = estimate_tokens(part, model=model)
                    if overlap_tok + pt > overlap_tokens:
                        break
                    overlap_parts.insert(0, part)
                    overlap_tok += pt
                current_parts = overlap_parts
                current_tokens = overlap_tok
            else:
                current_parts = []
                current_tokens = 0

        current_parts.append(text)
        current_tokens += text_tokens

    if current_parts:
        chunks.append(
            {
                "text": "\n\n".join(current_parts),
                "token_count": current_tokens,
            }
        )

    return chunks


def chunk_rag_context(
    elements: List[Dict[str, Any]],
    max_tokens: int = 400,
    overlap_tokens: int = 50,
    model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Hierarchical, token-aware chunking of parsed document elements with metadata preservation.

    Automatically selects strategy based on document structure:
      - structured: section → paragraph → sentence hierarchy
      - flat_narrative: synthetic paragraph grouping → sentence fallback
      - atomic_flat: each atomic element = own chunk; narrative grouped separately
      - monolithic: sentence splitting → merge up to budget
      - mixed: anchored sections + flat fallback for unanchored runs

    Args:
        elements: List of element dicts from partition().to_dict()
        max_tokens: Target max tokens per chunk. Pass None to auto-derive from model.
        overlap_tokens: Overlap between consecutive chunks (0 for sparse retrieval)
        model: Optional llama.cpp model key for accurate token counting and
               auto chunk sizing. Uses default LLM_MODEL if None and max_tokens is None.

    Returns:
        List of chunk dicts with 'text', 'token_count', 'strategy', and 'metadata' keys.
    """
    if not elements:
        return []

    # ✅ Auto-derive chunk size from model context if not explicitly set
    if max_tokens is None:
        max_tokens = auto_chunk_size(model)
        logger.info(
            f"chunk_rag_context | auto-derived max_tokens={max_tokens} from model='{model}'"
        )

    rag_elements = [
        e
        for e in elements
        if e.get("type") in RAG_CONTEXT_TYPES and e.get("text", "").strip()
    ]

    if not rag_elements:
        logger.warning(
            "chunk_rag_context | No valid RAG elements found after filtering"
        )
        return []

    structure = classify_structure(rag_elements)
    logger.info(
        f"chunk_rag_context | structure={structure} | elements={len(rag_elements)} | max_tokens={max_tokens}"
    )

    def _build_chunk(
        text: str, token_count: int, strategy: str, source_elems: List[Dict]
    ) -> Dict[str, Any]:
        """Build a chunk dict with aggregated metadata from source elements."""
        primary_meta = source_elems[0].get("metadata", {}) if source_elems else {}
        element_ids = [e.get("element_id") for e in source_elems if e.get("element_id")]
        return {
            "text": text,
            "token_count": token_count,
            "strategy": strategy,
            "metadata": {
                "filename": primary_meta.get("filename"),
                "page_number": primary_meta.get("page_number"),
                "languages": primary_meta.get("languages"),
                "element_ids": element_ids,
                "parent_id": primary_meta.get("parent_id"),
            },
        }

    chunks: List[Dict[str, Any]] = []

    if structure == "structured":
        sections: List[List[Dict]] = [[]]
        for elem in rag_elements:
            if elem.get("type") in SECTION_TYPES and sections[-1]:
                sections.append([])
            sections[-1].append(elem)

        for section_elems in sections:
            if not section_elems:
                continue
            narrative_parts: List[str] = []
            narrative_sources: List[Dict] = []

            for elem in section_elems:
                etype = elem.get("type", "")
                text = elem.get("text", "").strip()

                if etype in ATOMIC_TYPES:
                    if narrative_parts:
                        merged_chunks = _merge_up_to_budget(
                            narrative_parts, max_tokens, overlap_tokens, model=model
                        )
                        for mc in merged_chunks:
                            chunks.append(
                                _build_chunk(
                                    mc["text"],
                                    mc["token_count"],
                                    "structured",
                                    narrative_sources,
                                )
                            )
                        narrative_parts = []
                        narrative_sources = []

                    tok = estimate_tokens(text, model=model)
                    chunks.append(_build_chunk(text, tok, "atomic", [elem]))
                else:
                    narrative_parts.append(text)
                    narrative_sources.append(elem)

            if narrative_parts:
                merged_chunks = _merge_up_to_budget(
                    narrative_parts, max_tokens, overlap_tokens, model=model
                )
                for mc in merged_chunks:
                    chunks.append(
                        _build_chunk(
                            mc["text"],
                            mc["token_count"],
                            "structured",
                            narrative_sources,
                        )
                    )

    elif structure == "atomic_flat":
        narrative_parts: List[str] = []
        narrative_sources: List[Dict] = []

        for elem in rag_elements:
            etype = elem.get("type", "")
            text = elem.get("text", "").strip()

            if etype in ATOMIC_TYPES:
                if narrative_parts:
                    merged_chunks = _merge_up_to_budget(
                        narrative_parts, max_tokens, overlap_tokens, model=model
                    )
                    for mc in merged_chunks:
                        chunks.append(
                            _build_chunk(
                                mc["text"],
                                mc["token_count"],
                                "atomic_flat",
                                narrative_sources,
                            )
                        )
                    narrative_parts = []
                    narrative_sources = []

                chunks.append(
                    _build_chunk(
                        text, estimate_tokens(text, model=model), "atomic", [elem]
                    )
                )
            else:
                narrative_parts.append(text)
                narrative_sources.append(elem)

        if narrative_parts:
            merged_chunks = _merge_up_to_budget(
                narrative_parts, max_tokens, overlap_tokens, model=model
            )
            for mc in merged_chunks:
                chunks.append(
                    _build_chunk(
                        mc["text"], mc["token_count"], "atomic_flat", narrative_sources
                    )
                )

    elif structure == "flat_narrative":
        GROUP_SIZE = 4
        texts = [e.get("text", "").strip() for e in rag_elements]
        sources = rag_elements

        for i in range(0, len(texts), GROUP_SIZE):
            group_texts = texts[i : i + GROUP_SIZE]
            group_sources = sources[i : i + GROUP_SIZE]
            merged = "\n\n".join(group_texts)

            if estimate_tokens(merged, model=model) <= max_tokens:
                chunks.append(
                    _build_chunk(
                        merged,
                        estimate_tokens(merged, model=model),
                        "synthetic_para",
                        group_sources,
                    )
                )
            else:
                sentences: List[str] = []
                sent_sources: List[Dict] = []
                for t, src in zip(group_texts, group_sources):
                    sents = split_sentences(t)
                    sentences.extend(sents)
                    sent_sources.extend([src] * len(sents))

                merged_chunks = _merge_up_to_budget(
                    sentences, max_tokens, overlap_tokens, model=model
                )
                for mc in merged_chunks:
                    chunks.append(
                        _build_chunk(
                            mc["text"],
                            mc["token_count"],
                            "flat_narrative_split",
                            group_sources,
                        )
                    )

    elif structure == "monolithic":
        text = rag_elements[0].get("text", "").strip()
        sentences = split_sentences(text)

        if len(sentences) == 1 and estimate_tokens(text, model=model) <= max_tokens:
            chunks.append(
                _build_chunk(
                    text,
                    estimate_tokens(text, model=model),
                    "monolithic",
                    [rag_elements[0]],
                )
            )
        else:
            merged_chunks = _merge_up_to_budget(
                sentences, max_tokens, overlap_tokens, model=model
            )
            for mc in merged_chunks:
                chunks.append(
                    _build_chunk(
                        mc["text"],
                        mc["token_count"],
                        "monolithic_split",
                        [rag_elements[0]],
                    )
                )

    else:  # mixed
        anchored_run: List[Dict] = []
        flat_run: List[str] = []
        flat_sources: List[Dict] = []

        for elem in rag_elements:
            if elem.get("type") in SECTION_TYPES:
                if flat_run:
                    merged_chunks = _merge_up_to_budget(
                        flat_run, max_tokens, overlap_tokens, model=model
                    )
                    for mc in merged_chunks:
                        chunks.append(
                            _build_chunk(
                                mc["text"],
                                mc["token_count"],
                                "mixed_flat",
                                flat_sources,
                            )
                        )
                    flat_run = []
                    flat_sources = []
                anchored_run = [elem]
            elif anchored_run:
                anchored_run.append(elem)
            else:
                flat_run.append(elem.get("text", "").strip())
                flat_sources.append(elem)

        if anchored_run:
            narrative_parts: List[str] = []
            narrative_sources: List[Dict] = []

            for elem in anchored_run:
                etype = elem.get("type", "")
                text = elem.get("text", "").strip()

                if etype in ATOMIC_TYPES:
                    if narrative_parts:
                        merged_chunks = _merge_up_to_budget(
                            narrative_parts, max_tokens, overlap_tokens, model=model
                        )
                        for mc in merged_chunks:
                            chunks.append(
                                _build_chunk(
                                    mc["text"],
                                    mc["token_count"],
                                    "mixed_anchored",
                                    narrative_sources,
                                )
                            )
                        narrative_parts = []
                        narrative_sources = []

                    chunks.append(
                        _build_chunk(
                            text, estimate_tokens(text, model=model), "atomic", [elem]
                        )
                    )
                else:
                    narrative_parts.append(text)
                    narrative_sources.append(elem)

            if narrative_parts:
                merged_chunks = _merge_up_to_budget(
                    narrative_parts, max_tokens, overlap_tokens, model=model
                )
                for mc in merged_chunks:
                    chunks.append(
                        _build_chunk(
                            mc["text"],
                            mc["token_count"],
                            "mixed_anchored",
                            narrative_sources,
                        )
                    )

        if flat_run:
            merged_chunks = _merge_up_to_budget(
                flat_run, max_tokens, overlap_tokens, model=model
            )
            for mc in merged_chunks:
                chunks.append(
                    _build_chunk(
                        mc["text"], mc["token_count"], "mixed_flat", flat_sources
                    )
                )

    for c in chunks:
        c.setdefault("strategy", structure)

    total_tokens = sum(c["token_count"] for c in chunks)
    strategies_used = sorted({c["strategy"] for c in chunks})
    logger.info(
        f"chunk_rag_context | produced {len(chunks)} chunks | "
        f"total_tokens={total_tokens} | strategies={strategies_used}"
    )

    return chunks


def _parse_notebook(path: str) -> List[Dict[str, Any]]:
    """Parse .ipynb natively into unstructured-compatible element dicts."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except Exception as e:
        logger.error(f"_parse_notebook | FAILED | path={path} | error={e}")
        return []

    elements: List[Dict[str, Any]] = []
    cells = nb.get("cells", [])
    kernel_lang = nb.get("metadata", {}).get("language_info", {}).get("name", "")

    for cell in cells:
        cell_type = cell.get("cell_type", "")
        source_lines = cell.get("source", [])
        text = "".join(source_lines).strip()
        if not text:
            continue

        if cell_type == "markdown":
            for line in text.split("\n"):
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("#"):
                    elements.append(
                        {
                            "type": "Title",
                            "text": stripped.lstrip("#").strip(),
                            "metadata": {},
                        }
                    )
                elif stripped.startswith("- ") or stripped.startswith("* "):
                    elements.append(
                        {
                            "type": "ListItem",
                            "text": stripped[2:].strip(),
                            "metadata": {},
                        }
                    )
                else:
                    elements.append(
                        {"type": "NarrativeText", "text": stripped, "metadata": {}}
                    )
        elif cell_type == "code":
            elements.append(
                {
                    "type": "CodeSnippet",
                    "text": text,
                    "metadata": {"language": kernel_lang},
                }
            )

    logger.info(f"_parse_notebook | parsed {len(elements)} elements from {path}")
    return elements


def extract_rag_context(elements: List[Dict[str, Any]]) -> str:
    """Extract clean RAG-ready text from parsed elements."""
    if not elements:
        return ""

    rag_parts: List[str] = []
    for elem in elements:
        elem_type = elem.get("type", "")
        text = elem.get("text", "").strip()

        if elem_type in RAG_CONTEXT_TYPES and text:
            if elem_type == "Table":
                rag_parts.append(f"[TABLE]\n{text}\n[/TABLE]")
            elif elem_type == "CodeSnippet":
                lang = elem.get("metadata", {}).get("language", "")
                header = f"[CODE{f' ({lang})' if lang else ''}]"
                rag_parts.append(f"{header}\n{text}\n[/CODE]")
            elif elem_type == "Formula":
                rag_parts.append(f"[FORMULA]{text}[/FORMULA]")
            else:
                rag_parts.append(text)

    context = "\n\n".join(rag_parts)
    logger.info(
        f"extract_rag_context | kept={len(rag_parts)} elements | "
        f"chars={len(context)} | types_used="
        f"{sorted({e.get('type') for e in elements if e.get('type') in RAG_CONTEXT_TYPES})}"
    )
    return context


def _extract_document_metadata(
    path: str,
) -> tuple[List[Dict[str, Any]], List[str], int, Optional[int]]:
    """
    Parse a document and extract raw elements plus summary metadata.

    Handles .ipynb files natively and delegates other formats to
    unstructured.partition.auto. Supports both local files and URLs.

    Args:
        path: File path or URL to parse.

    Returns:
        Tuple of (element_dicts, categories, word_count, page_count).
    """
    if path.endswith(".ipynb"):
        element_dicts = _parse_notebook(path)
        categories = sorted({e.get("type", "Unknown") for e in element_dicts})
        full_text = " ".join(e.get("text", "") for e in element_dicts)
        word_count = len(full_text.split())
        page_count = None
    else:
        if path.startswith(("http://", "https://")):
            elements = partition(url=path)
        else:
            elements = partition(filename=path)

        categories = [getattr(e, "category", "Unknown") for e in elements]
        full_text = " ".join(str(e) for e in elements)
        word_count = len(full_text.split())
        page_numbers = [
            getattr(e.metadata, "page_number", None)
            for e in elements
            if hasattr(e, "metadata")
        ]
        page_count = max((p for p in page_numbers if p is not None), default=None)
        element_dicts = [e.to_dict() for e in elements]

    return element_dicts, categories, word_count, page_count


def parse_document(
    path: str,
    chunk_max_tokens: Optional[int] = None,
    chunk_overlap_tokens: int = 50,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Parse any document and produce RAG-ready chunks.

    Handles local files, URLs, notebooks, and unsupported types gracefully.

    Args:
        path: File path or URL to parse.
        chunk_max_tokens: Max tokens per chunk. Pass None to auto-derive from
                          the target model's context window (recommended).
                          Explicit values override auto-sizing.
        chunk_overlap_tokens: Overlap tokens between consecutive chunks.
        model: llama.cpp model key (e.g., "qwen3.5:2b") for accurate token
               counting and auto chunk sizing. Uses default LLM_MODEL if None.
    """
    logger.info(f"parse_document | START | path={path} | model={model}")
    try:
        element_dicts, categories, word_count, page_count = _extract_document_metadata(
            path
        )

        rag_context = extract_rag_context(element_dicts)
        chunks = chunk_rag_context(
            element_dicts,
            max_tokens=chunk_max_tokens,
            overlap_tokens=chunk_overlap_tokens,
            model=model,
        )

        result = {
            "path": path,
            "element_count": len(element_dicts),
            "categories": sorted(set(categories)),
            "word_count": word_count,
            "page_count": page_count,
            "elements": element_dicts,
            "rag_context": rag_context,
            "chunks": chunks,
            "status": "success",
        }
        logger.info(
            f"parse_document | DONE | elements={len(element_dicts)} | "
            f"words={word_count} | pages={page_count} | "
            f"chunks={len(chunks)} | categories={result['categories']}"
        )
        return result

    except Exception as e:
        logger.error(
            f"parse_document | FAILED | path={path} | error={e}", exc_info=True
        )
        return {
            "path": path,
            "element_count": 0,
            "categories": [],
            "word_count": 0,
            "page_count": None,
            "elements": [],
            "rag_context": "",
            "chunks": [],
            "status": f"error: {e}",
        }
