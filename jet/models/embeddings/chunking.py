"""
Hierarchical Markdown Chunking Module

Splits markdown documents into semantically-aware chunks that preserve header
hierarchy, parent-child relationships, and precise source-text indices. Each
chunk carries metadata enabling reconstruction of the original document structure
for retrieval-augmented generation (RAG) pipelines.

Key behaviors:
- Chunks respect a configurable token budget (header tokens count toward limit).
- Parent/child header relationships are tracked via generated IDs.
- Source text indices (`start_idx`, `end_idx`) include the header line,
  enabling full semantic-unit reconstruction from a single slice.
- Headers with no body content before the next header are skipped (no empty chunks).
- Sentences exceeding the chunk size are emitted as a single oversized chunk.
- All chunks under the same header consistently include the header in their
  metadata range, even when split by token budget.

Minimal Usage Examples
----------------------

Single document:

    >>> from jet.models.embeddings.chunking import chunk_headers_by_hierarchy
    >>> text = '''
    ... # Introduction
    ... Welcome to the guide.
    ... ## Setup
    ... Install dependencies first.
    ... Then configure your environment.
    ... '''
    >>> chunks = chunk_headers_by_hierarchy(text, chunk_size=10)
    >>> len(chunks)
    2
    >>> chunks[0]["header"]
    '# Introduction'
    >>> chunks[0]["content"]
    'Welcome to the guide.'
    >>> chunks[1]["parent_header"]
    '# Introduction'
    >>> sliced = text[chunks[1]["metadata"]["start_idx"]:chunks[1]["metadata"]["end_idx"]].strip()
    >>> sliced.startswith(chunks[1]["header"])
    True
    >>> chunks[1]["content"] in sliced
    True

Multiple documents with explicit IDs:

    >>> from jet.models.embeddings.chunking import chunk_docs_by_hierarchy
    >>> docs = ["# Doc A\\nContent A.", "# Doc B\\nContent B."]
    >>> results = chunk_docs_by_hierarchy(docs, chunk_size=10, ids=["a1", "b2"])
    >>> {r["doc_id"] for r in results}
    {'a1', 'b2'}
"""

import re
from typing import Callable, List, Optional, TypedDict, Union

from jet.data.utils import generate_unique_id
from jet.models.tokenizer.base import TokenizerWrapper


class ChunkMetadata(TypedDict):
    start_idx: int
    end_idx: int


class ChunkResult(TypedDict):
    id: str
    parent_id: Optional[str]
    header_doc_id: str
    doc_index: int
    chunk_index: int
    num_tokens: int
    header: str
    parent_header: Optional[str]
    content: str
    level: int
    parent_level: Optional[int]
    metadata: ChunkMetadata


def chunk_headers_by_hierarchy(
    markdown_text: str,
    chunk_size: int,
    tokenizer: Optional[
        Union[
            Callable[[Union[str, List[str]]], Union[List[str], List[List[str]]]],
            TokenizerWrapper,
        ]
    ] = None,
    split_fn: Optional[Callable[[str], List[str]]] = None,
) -> List[ChunkResult]:
    """
    Chunk a single markdown document by header hierarchy.

    Args:
        markdown_text: Raw markdown string to chunk.
        chunk_size: Maximum tokens per chunk (including header tokens).
        tokenizer: Token counting function or TokenizerWrapper.
                   Defaults to nltk.word_tokenize.
        split_fn: Sentence splitting function. Defaults to nltk.sent_tokenize.

    Returns:
        List of ChunkResult dicts with content, hierarchy metadata, and
        source-text indices. Empty input returns an empty list.
    """
    if tokenizer is None:
        import nltk

        def tokenizer(x):
            return (
                nltk.word_tokenize(x)
                if isinstance(x, str)
                else [nltk.word_tokenize(t) for t in x]
            )

    if split_fn is None:
        import nltk

        split_fn = nltk.sent_tokenize

    if not markdown_text.strip():
        return []

    HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    results: List[ChunkResult] = []
    header_stack: List[dict] = []  # {level, text, header_doc_id}
    section_counter = 0

    current = {
        "content_parts": [],
        "token_count": 0,
        "header": "",
        "header_tokens": 0,
        "level": 0,
        "parent_header": None,
        "parent_level": None,
        "parent_id": None,
        "header_doc_id": "",
        "section_index": 0,
        "chunk_index": 0,
        "_header_abs_start": 0,
        "abs_start": 0,
        "abs_end": 0,
    }

    def _resolve_parent(level: int):
        """Find nearest ancestor with lower level in single pass."""
        for h in reversed(header_stack):
            if h["level"] < level:
                return h["text"], h["level"], h["header_doc_id"]
        return None, None, None

    def _flush_chunk():
        if not current["content_parts"]:
            return

        content_str = "\n".join(current["content_parts"]).strip()
        total_tokens = current["token_count"] + current["header_tokens"]

        chunk: ChunkResult = {
            "id": generate_unique_id(),
            "parent_id": current["parent_id"],
            "header_doc_id": current["header_doc_id"],
            "doc_index": current["section_index"],
            "chunk_index": current["chunk_index"],
            "num_tokens": total_tokens,
            "header": current["header"],
            "parent_header": current["parent_header"],
            "content": content_str,
            "level": current["level"],
            "parent_level": current["parent_level"],
            "metadata": {
                "start_idx": current["abs_start"],
                "end_idx": current["abs_end"],
            },
        }
        results.append(chunk)
        current["chunk_index"] += 1
        current["content_parts"] = []
        current["token_count"] = 0

        # Reset BOTH anchors so next chunk starts fresh at header position
        current["abs_start"] = current["_header_abs_start"]
        current["abs_end"] = current["_header_abs_start"]  # ✅ ADD THIS LINE

    def _count_tokens(text: str) -> int:
        tokens = tokenizer(text)
        if isinstance(tokenizer, TokenizerWrapper):
            return len(tokens._ids) if hasattr(tokens, "_ids") else len(tokens)
        if isinstance(tokens, list):
            return len(tokens)
        return 0

    pos = 0
    lines = markdown_text.split("\n")

    for line in lines:
        line_len = len(line) + 1  # +1 for \n
        stripped = line.strip()

        header_match = HEADER_RE.match(stripped)
        if header_match:
            _flush_chunk()
            section_counter += 1

            level = len(header_match.group(1))
            header_text = stripped

            # Update hierarchy stack
            header_stack = [h for h in header_stack if h["level"] < level]
            header_doc_id = generate_unique_id()
            header_stack.append(
                {
                    "level": level,
                    "text": header_text,
                    "header_doc_id": header_doc_id,
                }
            )

            parent_header, parent_level, parent_id = _resolve_parent(level)
            hdr_token_count = _count_tokens(header_text)

            current.update(
                {
                    "header": header_text,
                    "header_tokens": hdr_token_count,
                    "level": level,
                    "parent_header": parent_header,
                    "parent_level": parent_level,
                    "parent_id": parent_id,
                    "header_doc_id": header_doc_id,
                    "section_index": section_counter,
                    "chunk_index": 0,
                    "_header_abs_start": pos,
                    "abs_start": pos,
                    "abs_end": pos + len(line),
                }
            )

            pos += line_len
            continue

        # Process body content
        if stripped:
            sentences = split_fn(line)
            search_offset = 0

            for sent in sentences:
                sent_stripped = sent.strip()
                if not sent_stripped:
                    continue

                # Find exact position of this sentence instance in the line
                idx = line.find(sent_stripped, search_offset)
                if idx == -1:
                    idx = search_offset

                abs_sent_start = pos + idx
                abs_sent_end = abs_sent_start + len(sent_stripped)

                sent_token_count = _count_tokens(sent_stripped)

                # Budget check: header + accumulated + new sentence
                projected = (
                    current["token_count"] + sent_token_count + current["header_tokens"]
                )

                if projected > chunk_size and current["content_parts"]:
                    _flush_chunk()

                if not current["content_parts"]:
                    current["abs_start"] = current["_header_abs_start"]

                current["content_parts"].append(sent_stripped)
                current["token_count"] += sent_token_count
                current["abs_end"] = abs_sent_end

                search_offset = idx + len(sent_stripped)

        pos += line_len

    _flush_chunk()
    return results


class DocChunkResult(ChunkResult):
    doc_id: str


def chunk_docs_by_hierarchy(
    markdown_texts: List[str],
    chunk_size: int,
    tokenizer: Optional[
        Union[
            Callable[[Union[str, List[str]]], Union[List[str], List[List[str]]]],
            TokenizerWrapper,
        ]
    ] = None,
    split_fn: Optional[Callable[[str], List[str]]] = None,
    ids: Optional[List[str]] = None,
) -> List[DocChunkResult]:
    """
    Chunk multiple markdown documents by hierarchy, preserving document IDs.

    Args:
        markdown_texts: List of markdown text strings to chunk.
        chunk_size: Maximum number of tokens per chunk.
        tokenizer: Optional tokenizer function or TokenizerWrapper.
        split_fn: Optional sentence splitting function.
        ids: Optional list of document IDs; if None, generates unique IDs.

    Returns:
        List of DocChunkResult dictionaries containing chunked content with
        document IDs.

    Raises:
        ValueError: If len(ids) != len(markdown_texts).
    """
    if tokenizer is None:
        import nltk

        def tokenizer(x):
            return (
                nltk.word_tokenize(x)
                if isinstance(x, str)
                else [nltk.word_tokenize(t) for t in x]
            )

    if split_fn is None:
        import nltk

        split_fn = nltk.sent_tokenize

    results: List[DocChunkResult] = []
    doc_ids = ids if ids else [generate_unique_id() for _ in markdown_texts]
    if len(doc_ids) != len(markdown_texts):
        raise ValueError("Number of provided IDs must match number of documents")

    for doc_idx, (markdown_text, doc_id) in enumerate(zip(markdown_texts, doc_ids)):
        doc_chunks = chunk_headers_by_hierarchy(
            markdown_text, chunk_size, tokenizer, split_fn
        )
        for chunk in doc_chunks:
            chunk_result: DocChunkResult = {
                **chunk,
                "doc_id": doc_id,
                "doc_index": doc_idx,
            }
            results.append(chunk_result)

    return results
