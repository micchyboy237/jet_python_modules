from typing import Dict, TypedDict, Callable, Union, List, Optional
import re
import nltk
from jet.data.utils import generate_unique_id
from jet.models.tokenizer.base import TokenizerWrapper, EncodingWrapper


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
    tokenizer: Optional[Union[
        Callable[[Union[str, List[str]]], Union[List[str], List[List[str]]]],
        TokenizerWrapper
    ]] = None,
    split_fn: Optional[Callable[[str], List[str]]] = None
) -> List[ChunkResult]:
    if tokenizer is None:
        def tokenizer(x): return nltk.word_tokenize(x) if isinstance(
            x, str) else [nltk.word_tokenize(t) for t in x]
    split_fn = split_fn or nltk.sent_tokenize
    if not markdown_text.strip():
        return []
    lines = markdown_text.strip().split('\n')
    header_pattern = r'^(#{1,6})\s+(.+)$'
    results = []
    current = {
        "id": None,
        "parent_id": None,
        "header_doc_id": None,
        "content": [],
        "num_tokens": 0,
        "header": "",
        "parent_header": None,
        "level": 0,
        "parent_level": None,
        "doc_index": 0,
        "chunk_index": 0,
        "start_idx": 0,
        "end_idx": 0,
        "metadata": {},
    }
    header_stack = []
    doc_index = -1
    char_index = 0

    def add_chunk():
        if current["content"]:
            header_tokens = (
                tokenizer(current["header"])._ids
                if isinstance(tokenizer, TokenizerWrapper)
                else tokenizer(current["header"])
            ) if current["header"] else []
            content_str = "\n".join(current["content"]).strip()
            chunk = {
                "id": generate_unique_id(),
                "parent_id": current["parent_id"],
                "header_doc_id": current["header_doc_id"],
                "doc_index": current["doc_index"],
                "chunk_index": current["chunk_index"],
                "num_tokens": current["num_tokens"] + (len(header_tokens) if isinstance(header_tokens, list) else 0),
                "header": current["header"],
                "parent_header": current["parent_header"],
                "content": content_str,
                "level": current["level"],
                "parent_level": current["parent_level"],
                "metadata": {
                    "start_idx": current["start_idx"],
                    "end_idx": current["end_idx"]
                }
            }
            results.append(chunk)
            current["chunk_index"] += 1
            current["content"] = []
            current["num_tokens"] = 0

    if lines and not lines[0].strip():
        char_index += len(lines[0]) + 1
        lines = lines[1:]

    for line in lines:
        line_with_newline = line + '\n'
        line = line.strip()
        if not line:
            char_index += len(line_with_newline)
            continue
        header_match = re.match(header_pattern, line, re.MULTILINE)
        if header_match:
            add_chunk()
            doc_index += 1
            current["level"] = len(header_match.group(1))
            current["header"] = header_match.group(0).strip()
            header_stack = [
                h for h in header_stack if h["level"] < current["level"]]
            header_stack.append({
                "level": current["level"],
                "text": current["header"],
                "header_doc_id": generate_unique_id()  # Store header_doc_id in header_stack
            })
            current["parent_header"] = next(
                (h["text"] for h in header_stack[::-1]
                 if h["level"] < current["level"]), None
            ) if current["level"] > 1 else None
            current["parent_level"] = next(
                (h["level"] for h in header_stack[::-1]
                 if h["level"] < current["level"]), None
            ) if current["level"] > 1 else None
            current["parent_id"] = next(
                (h["header_doc_id"] for h in header_stack[::-1]
                 if h["level"] < current["level"]), None
            ) if current["level"] > 1 else None  # Assign parent_id
            current["header_doc_id"] = header_stack[-1]["header_doc_id"]
            current["doc_index"] = doc_index
            current["chunk_index"] = 0
            char_index += len(line_with_newline)
            current["start_idx"] = char_index
            continue

        sentences = split_fn(line) if split_fn else [line]
        for sentence_idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                char_index += len(line_with_newline)
                continue

            tokens = (
                tokenizer(sentence)._ids
                if isinstance(tokenizer, TokenizerWrapper)
                else tokenizer(sentence)
            )
            num_tokens = len(tokens) if isinstance(
                tokens, list) else sum(len(t) for t in tokens)
            header_tokens = (
                tokenizer(current["header"])._ids
                if isinstance(tokenizer, TokenizerWrapper)
                else tokenizer(current["header"])
            ) if current["header"] else []
            header_num_tokens = len(header_tokens) if isinstance(
                header_tokens, list) else 0

            if not current["content"]:
                current["start_idx"] = char_index + 1

            if current["num_tokens"] + num_tokens + header_num_tokens <= chunk_size:
                current["content"].append(sentence)
                current["num_tokens"] += num_tokens
                current["end_idx"] = char_index + len(sentence)
            else:
                add_chunk()
                current["content"] = [sentence]
                current["num_tokens"] = num_tokens
                current["start_idx"] = char_index + 1
                current["end_idx"] = char_index + len(sentence)

            char_index += len(sentence)
            if sentence_idx < len(sentences) - 1:
                char_index += 1

        char_index += 1

    add_chunk()
    return results


class DocChunkResult(ChunkResult):
    doc_id: str


def chunk_docs_by_hierarchy(
    markdown_texts: List[str],
    chunk_size: int,
    tokenizer: Optional[Union[
        Callable[[Union[str, List[str]]], Union[List[str], List[List[str]]]],
        TokenizerWrapper
    ]] = None,
    split_fn: Optional[Callable[[str], List[str]]] = None,
    ids: Optional[List[str]] = None
) -> List[DocChunkResult]:
    """
    Chunks a list of markdown documents by hierarchy, preserving document IDs.

    Args:
        markdown_texts: List of markdown text strings to chunk.
        chunk_size: Maximum number of tokens per chunk.
        tokenizer: Optional tokenizer function or TokenizerWrapper.
        split_fn: Optional sentence splitting function.
        ids: Optional list of document IDs; if None, generates unique IDs.

    Returns:
        List of DocChunkResult dictionaries containing chunked content with document IDs.
    """
    if tokenizer is None:
        def tokenizer(x): return nltk.word_tokenize(x) if isinstance(
            x, str) else [nltk.word_tokenize(t) for t in x]
    split_fn = split_fn or nltk.sent_tokenize
    results: List[DocChunkResult] = []
    doc_ids = ids if ids else [generate_unique_id() for _ in markdown_texts]

    if len(doc_ids) != len(markdown_texts):
        raise ValueError(
            "Number of provided IDs must match number of documents")

    for doc_idx, (markdown_text, doc_id) in enumerate(zip(markdown_texts, doc_ids)):
        # Get chunks for the current document using chunk_headers_by_hierarchy
        doc_chunks = chunk_headers_by_hierarchy(
            markdown_text, chunk_size, tokenizer, split_fn
        )
        # Update each chunk with the document ID and adjust doc_index to be unique across all documents
        for chunk in doc_chunks:
            chunk_result: DocChunkResult = {
                **chunk,
                "doc_id": doc_id,
                "doc_index": doc_idx  # Override doc_index to reflect document position in input list
            }
            results.append(chunk_result)

    return results
