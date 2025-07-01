import uuid
from typing import List, Optional, Union, Literal
from pathlib import Path
from pydantic import BaseModel, Field
from jet.code.markdown_types import MarkdownToken, ContentType, MetaType
from jet.data.utils import generate_unique_id
from jet.models.tokenizer.base import get_tokenizer, tokenize, detokenize
from jet.models.model_types import ModelType
from jet.data.header_types import Node, HeaderNode, TextNode, NodeType, Nodes
from tokenizers import Tokenizer
from jet.logger import logger


def create_text_node(
    node: NodeType,
    content: str,
    chunk_index: int,
    parent_id: Optional[str] = None,
    parent_header: Optional[str] = None
) -> TextNode:
    """Create a new TextNode with the given parameters."""
    new_node = TextNode(
        id=generate_unique_id(),
        line=node.line,
        parent_id=parent_id,
        parent_header=parent_header,
        type=node.type if isinstance(node, TextNode) else "paragraph",
        header=node.header,
        content=content.strip(),
        meta=node.meta if isinstance(node, TextNode) else None,
        chunk_index=chunk_index
    )
    if parent_id:
        new_node._parent_node = node.get_parent_node()
    return new_node


def chunk_content(
    content: str,
    tokenizer: Optional[Tokenizer],
    chunk_size: Optional[int],
    chunk_overlap: int,
    buffer: int
) -> List[str]:
    """Split content into chunks based on token count."""
    if not content or not chunk_size or not tokenizer:
        return [content]

    token_ids = tokenize(content, tokenizer, add_special_tokens=False)
    if isinstance(token_ids[0], list):
        token_ids = token_ids[0]

    effective_chunk_size = chunk_size - buffer
    if len(token_ids) <= effective_chunk_size:
        return [content]

    chunks = []
    start = 0
    step = effective_chunk_size - chunk_overlap
    while start < len(token_ids):
        end = min(start + effective_chunk_size, len(token_ids))
        chunk_tokens = token_ids[start:end]
        chunk_text = detokenize(chunk_tokens, tokenizer)
        chunks.append(chunk_text)
        start += step
        if start >= len(token_ids) or (end == len(token_ids) and start + chunk_overlap >= len(token_ids)):
            break

    return chunks


def process_node(
    node: NodeType,
    tokenizer: Optional[Tokenizer],
    chunk_size: Optional[int],
    chunk_overlap: int,
    buffer: int,
    parent_id: Optional[str] = None,
    parent_header: Optional[str] = None
) -> List[TextNode]:
    """Process a single node and return a list of resulting TextNodes."""
    result_nodes: List[TextNode] = []
    content = node.content.strip()

    if not content and isinstance(node, TextNode):
        return [node]

    header_prefix = f"{node.header}\n" if node.header else ""
    chunks = chunk_content(
        content, tokenizer, chunk_size, chunk_overlap, buffer)

    for i, chunk in enumerate(chunks):
        full_content = f"{header_prefix}{chunk}" if chunk else header_prefix
        new_node = create_text_node(
            node, full_content, i, parent_id, parent_header)
        result_nodes.append(new_node)

    if isinstance(node, HeaderNode):
        for child in node.children:
            result_nodes.extend(
                process_node(child, tokenizer, chunk_size,
                             chunk_overlap, buffer, node.id, node.header)
            )

    return result_nodes


__all__ = [
    "process_node",
    "chunk_content",
    "create_text_node",
]
