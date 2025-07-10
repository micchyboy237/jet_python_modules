from typing import TypedDict, Callable, Union, List, Optional
import re
import nltk


class ChunkResult(TypedDict):
    content: str
    num_tokens: int
    header: str
    parent_header: Optional[str]
    level: int
    parent_level: Optional[int]
    doc_index: int
    chunk_index: int


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
        "chunk_index": 0
    }
    header_stack = []
    doc_index = -1
    is_first_sentence_under_level3 = False

    def add_chunk():
        if current["content"]:
            header_tokens = tokenizer(
                current["header"]) if current["header"] else []
            results.append({
                "content": "\n".join(current["content"]).strip(),
                "num_tokens": current["num_tokens"] + (len(header_tokens) if isinstance(header_tokens, list) else 0),
                "header": current["header"],
                "parent_header": current["parent_header"],
                "level": current["level"],
                "parent_level": current["parent_level"],
                "doc_index": current["doc_index"],
                "chunk_index": current["chunk_index"]
            })
            current["chunk_index"] += 1
            current["content"] = []
            current["num_tokens"] = 0

    for line in lines:
        line = line.strip()
        if not line:
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
            is_first_sentence_under_level3 = (
                current["level"] >= 3)
            continue
        sentences = split_fn(line) if split_fn else [line]
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            tokens = tokenizer(sentence)
            num_tokens = len(tokens) if isinstance(
                tokens, list) else sum(len(t) for t in tokens)
            header_tokens = tokenizer(
                current["header"]) if current["header"] else []
            header_num_tokens = len(header_tokens) if isinstance(
                header_tokens, list) else 0
            if current["level"] >= 3:
                add_chunk()
                if not is_first_sentence_under_level3:
                    doc_index += 1
                current["doc_index"] = doc_index
                current["chunk_index"] = 0
                current["content"] = [sentence]
                current["num_tokens"] = num_tokens
                add_chunk()
                is_first_sentence_under_level3 = False
            elif current["num_tokens"] + num_tokens + header_num_tokens <= chunk_size:
                current["content"].append(sentence)
                current["num_tokens"] += num_tokens
            else:
                add_chunk()
                current["content"] = [sentence]
                current["num_tokens"] = num_tokens
    add_chunk()
    return results


class MergedChunkResult(TypedDict):
    content: str
    level: int
    headers: List[str]
    header: str
    parent_header: Optional[str]
    parent_level: Optional[int]
    num_tokens: int
    doc_index: int
    chunk_count: int


def merge_same_level_chunks(
    chunks: List[ChunkResult],
    chunk_size: int,
    tokenizer: Optional[Callable[[Union[str, List[str]]],
                                 Union[List[str], List[List[str]]]]] = None
) -> List[MergedChunkResult]:
    # Set default tokenizer if None
    tokenizer = tokenizer or (lambda x: nltk.word_tokenize(
        x) if isinstance(x, str) else [nltk.word_tokenize(t) for t in x])

    if not chunks:
        return []
    results: List[MergedChunkResult] = []
    current: Optional[MergedChunkResult] = None
    for chunk in chunks:
        content_with_header = f"{chunk['header']}\n{chunk['content']}" if chunk["header"] else chunk["content"]
        tokens = tokenizer(content_with_header)
        num_tokens = sum(1 for t in tokens if t.isalnum())
        if not current or current["level"] != chunk["level"]:
            if current:
                results.append(current)
            current = {
                "content": content_with_header,
                "level": chunk["level"],
                "headers": [chunk["header"]] if chunk["header"] else [],
                "header": chunk["header"],
                "parent_header": chunk["parent_header"],
                "parent_level": chunk["parent_level"],
                "num_tokens": num_tokens,
                "doc_index": len(results),
                "chunk_count": 1
            }
        else:
            combined_content = f"{current['content']}\n\n{chunk['header']}\n{chunk['content']}" if chunk[
                "header"] else f"{current['content']}\n\n{chunk['content']}"
            combined_tokens = tokenizer(combined_content)
            combined_num_tokens = sum(
                1 for t in combined_tokens if t.isalnum())
            if combined_num_tokens <= chunk_size:
                current["content"] = combined_content
                current["num_tokens"] = combined_num_tokens
                current["chunk_count"] += 1
                if chunk["header"]:
                    current["headers"].append(chunk["header"])
                    current["header"] = "\n".join(current["headers"])
                current["parent_header"] = chunk["parent_header"]
                current["parent_level"] = chunk["parent_level"]
            else:
                results.append(current)
                current = {
                    "content": content_with_header,
                    "level": chunk["level"],
                    "headers": [chunk["header"]] if chunk["header"] else [],
                    "header": chunk["header"],
                    "parent_header": chunk["parent_header"],
                    "parent_level": chunk["parent_level"],
                    "num_tokens": num_tokens,
                    "doc_index": len(results),
                    "chunk_count": 1
                }
    if current:
        results.append(current)
    for i, result in enumerate(results):
        result["doc_index"] = i
    return results
