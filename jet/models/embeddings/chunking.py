from typing import TypedDict, Callable, Union, List, Optional
import re
import nltk


class ChunkResult(TypedDict):
    content: str
    num_tokens: int
    header: str
    parent_header: Optional[str]
    level: int
    doc_index: int
    chunk_index: int


def chunk_headers_by_hierarchy(
    markdown_text: str,
    chunk_size: int,
    tokenizer: Callable[[Union[str, List[str]]], Union[List[str], List[List[str]]]] = lambda x: nltk.word_tokenize(
        x) if isinstance(x, str) else [nltk.word_tokenize(t) for t in x],
    split_fn: Optional[Callable[[str], List[str]]] = nltk.sent_tokenize
) -> List[ChunkResult]:
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
        "doc_index": 0,
        "chunk_index": 0
    }
    header_stack = []
    doc_index = -1  # Initialize to -1 so first header gets doc_index 0
    # Track first sentence under level 3 header
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
            current["doc_index"] = doc_index
            current["chunk_index"] = 0
            is_first_sentence_under_level3 = (
                current["level"] >= 3)  # Set flag for level 3 header
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
                    doc_index += 1  # Increment for subsequent sentences only
                current["doc_index"] = doc_index
                current["chunk_index"] = 0
                current["content"] = [sentence]
                current["num_tokens"] = num_tokens
                add_chunk()
                is_first_sentence_under_level3 = False  # Reset after first sentence
            elif current["num_tokens"] + num_tokens + header_num_tokens <= chunk_size:
                current["content"].append(sentence)
                current["num_tokens"] += num_tokens
            else:
                add_chunk()
                current["content"] = [sentence]
                current["num_tokens"] = num_tokens

    add_chunk()
    return results
