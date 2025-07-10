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
