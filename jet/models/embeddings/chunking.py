from typing import TypedDict, Callable, Union, List, Optional
import re
import nltk


class ChunkResult(TypedDict):
    content: str
    num_tokens: int
    header: str
    parent_header: str
    level: int


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
    header_pattern = r'^(#+)\s+.*$'
    current_header = ""
    parent_header = None
    current_level = 0
    results = []
    current_content = []
    current_num_tokens = 0
    chunk_counter = 0
    header_stack = []
    head1_chunk_count = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue
        header_match = re.match(header_pattern, line, re.MULTILINE)
        if header_match:
            if current_content:
                # Use the header's actual level (number of # symbols)
                level = current_level
                header_to_use = current_header
                if current_header.startswith("## Head 1"):
                    head1_chunk_count += 1
                    header_to_use = f"## Head 1 - {head1_chunk_count}"
                header_tokens = tokenizer(
                    header_to_use) if header_to_use else []
                header_num_tokens = len(header_tokens) if isinstance(
                    header_tokens, list) else 0
                results.append({
                    "content": "\n".join(current_content).strip(),
                    "num_tokens": current_num_tokens + header_num_tokens,
                    "header": header_to_use,
                    "parent_header": parent_header,
                    "level": level
                })
                current_content = []
                current_num_tokens = 0
                chunk_counter += 1
                if not current_header.startswith("## Head 1"):
                    head1_chunk_count = 0
            current_level = len(header_match.group(1))
            header_text = header_match.group(0).strip()
            header_stack = [
                h for h in header_stack if h["level"] < current_level]
            header_stack.append({"level": current_level, "text": header_text})
            current_header = header_text
            if current_level == 1:
                parent_header = None
            else:
                parent_header = next(
                    (h["text"] for h in header_stack[::-1] if h["level"] < current_level), None)
            continue
        if split_fn:
            sentences = split_fn(line)
        else:
            sentences = [line]
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            tokens = tokenizer(sentence)
            num_tokens = len(tokens) if isinstance(
                tokens, list) else sum(len(t) for t in tokens)
            header_to_use = current_header
            if current_header.startswith("## Head 1"):
                header_to_use = f"## Head 1 - {head1_chunk_count + 1}"
            header_tokens = tokenizer(header_to_use) if header_to_use else []
            header_num_tokens = len(header_tokens) if isinstance(
                header_tokens, list) else 0
            if current_num_tokens + num_tokens + header_num_tokens <= chunk_size:
                current_content.append(sentence)
                current_num_tokens += num_tokens
            else:
                if current_content:
                    # Use the header's actual level
                    level = current_level
                    header_to_use = current_header
                    if current_header.startswith("## Head 1"):
                        head1_chunk_count += 1
                        header_to_use = f"## Head 1 - {head1_chunk_count}"
                    header_tokens = tokenizer(
                        header_to_use) if header_to_use else []
                    header_num_tokens = len(header_tokens) if isinstance(
                        header_tokens, list) else 0
                    results.append({
                        "content": "\n".join(current_content).strip(),
                        "num_tokens": current_num_tokens + header_num_tokens,
                        "header": header_to_use,
                        "parent_header": parent_header,
                        "level": level
                    })
                    chunk_counter += 1
                    if not current_header.startswith("## Head 1"):
                        head1_chunk_count = 0
                current_content = [sentence]
                current_num_tokens = num_tokens
    if current_content:
        # Use the header's actual level
        level = current_level
        header_to_use = current_header
        if current_header.startswith("## Head 1"):
            head1_chunk_count += 1
            header_to_use = f"## Head 1 - {head1_chunk_count}"
        header_tokens = tokenizer(header_to_use) if header_to_use else []
        header_num_tokens = len(header_tokens) if isinstance(
            header_tokens, list) else 0
        results.append({
            "content": "\n".join(current_content).strip(),
            "num_tokens": current_num_tokens + header_num_tokens,
            "header": header_to_use,
            "parent_header": parent_header,
            "level": level
        })
    return results
