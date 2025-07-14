from typing import Dict, TypedDict, Callable, Union, List, Optional
import re
import nltk
import logging

from jet.logger import logger


class Metadata(TypedDict):
    start_idx: int
    end_idx: int


class ChunkResult(TypedDict):
    content: str
    num_tokens: int
    header: str
    parent_header: Optional[str]
    level: int
    parent_level: Optional[int]
    doc_index: int
    chunk_index: int
    metadata: Metadata


def chunk_headers_by_hierarchy(
    markdown_text: str,
    chunk_size: int,
    tokenizer: Optional[Callable[[Union[str, List[str]]],
                                 Union[List[str], List[List[str]]]]] = None,
    split_fn: Optional[Callable[[str], List[str]]] = None
) -> List[ChunkResult]:
    # Set default tokenizer and split_fn if None
    tokenizer = tokenizer or (lambda x: nltk.word_tokenize(
        x) if isinstance(x, str) else [nltk.word_tokenize(t) for t in x])
    split_fn = split_fn or nltk.sent_tokenize

    if not markdown_text.strip():
        return []
    lines = markdown_text.strip().split('\n')
    header_pattern = r'^(#+)\s*(.*)'
    results = []
    current = {
        "content": [],
        "num_tokens": 0,
        "header": "",
        "parent_header": None,
        "level": 0,
        "parent_level": None,
        "doc_index": 0,
        "chunk_index": 0,
        "start_idx": 0,
        "end_idx": 0
    }
    header_stack = []
    doc_index = -1
    char_index = 0  # Tracks position in markdown_text

    def add_chunk():
        if current["content"]:
            header_tokens = tokenizer(
                current["header"]) if current["header"] else []
            content_str = "\n".join(current["content"]).strip()
            chunk = {
                "content": content_str,
                "num_tokens": current["num_tokens"] + (len(header_tokens) if isinstance(header_tokens, list) else 0),
                "header": current["header"],
                "parent_header": current["parent_header"],
                "level": current["level"],
                "parent_level": current["parent_level"],
                "doc_index": current["doc_index"],
                "chunk_index": current["chunk_index"],
                "metadata": {
                    "start_idx": current["start_idx"],
                    "end_idx": current["end_idx"]
                }
            }
            logger.debug(f"Adding chunk: {chunk}")
            results.append(chunk)
            current["chunk_index"] += 1
            current["content"] = []
            current["num_tokens"] = 0

    # Skip initial empty line
    if lines and not lines[0].strip():
        char_index += len(lines[0]) + 1  # Account for newline
        lines = lines[1:]
        logger.debug(f"Skipped initial empty line, char_index: {char_index}")

    for line in lines:
        line_with_newline = line + '\n'
        line = line.strip()
        logger.debug(f"Processing line: '{line}', char_index: {char_index}")
        if not line:
            char_index += len(line_with_newline)
            logger.debug(f"Empty line, updated char_index: {char_index}")
            continue
        header_match = re.match(header_pattern, line, re.MULTILINE)
        if header_match:
            add_chunk()
            doc_index += 1
            current["level"] = len(header_match.group(1))
            current["header"] = header_match.group(0).strip()
            header_stack = [
                h for h in header_stack if h["level"] < current["level"]]
            header_stack.append(
                {"level": current["level"], "text": current["header"]})
            current["parent_header"] = next(
                (h["text"] for h in header_stack[::-1]
                 if h["level"] < current["level"]), None
            ) if current["level"] > 1 else None
            current["parent_level"] = next(
                (h["level"] for h in header_stack[::-1]
                 if h["level"] < current["level"]), None
            ) if current["level"] > 1 else None
            current["doc_index"] = doc_index
            current["chunk_index"] = 0
            char_index += len(line_with_newline)
            current["start_idx"] = char_index  # Start after header
            logger.debug(
                f"Header '{current['header']}', start_idx: {current['start_idx']}, char_index: {char_index}")
            continue
        sentences = split_fn(line) if split_fn else [line]
        for sentence_idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                char_index += len(line_with_newline)
                logger.debug(
                    f"Empty sentence, updated char_index: {char_index}")
                continue
            tokens = tokenizer(sentence)
            num_tokens = len(tokens) if isinstance(
                tokens, list) else sum(len(t) for t in tokens)
            header_tokens = tokenizer(
                current["header"]) if current["header"] else []
            header_num_tokens = len(header_tokens) if isinstance(
                header_tokens, list) else 0
            if not current["content"]:  # First sentence in chunk
                current["start_idx"] = char_index + 1  # Skip leading newline
            if current["num_tokens"] + num_tokens + header_num_tokens <= chunk_size:
                current["content"].append(sentence)
                current["num_tokens"] += num_tokens
                current["end_idx"] = char_index + len(sentence)
            else:
                add_chunk()
                current["content"] = [sentence]
                current["num_tokens"] = num_tokens
                current["start_idx"] = char_index + 1  # Skip leading newline
                current["end_idx"] = char_index + len(sentence)
            char_index += len(sentence)
            if sentence_idx < len(sentences) - 1:
                char_index += 1  # Newline between sentences
            logger.debug(
                f"Sentence '{sentence}', start_idx: {current['start_idx']}, end_idx: {current['end_idx']}, char_index: {char_index}")
        char_index += 1  # Newline after line
    add_chunk()
    logger.debug(f"Final results: {results}")
    return results
