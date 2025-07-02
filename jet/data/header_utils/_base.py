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
        chunk_index=chunk_index,
        doc_id=node.doc_id  # Propagate required doc_id
    )
    if parent_id:
        new_node._parent_node = node.get_parent_node()
    return new_node


def chunk_content(
    content: str,
    tokenizer: Optional[Tokenizer],
    chunk_size: Optional[int],
    chunk_overlap: int,
    buffer: int,
    header_prefix: str = ""
) -> List[str]:
    """Split content into chunks based on token count, preserving original case."""
    if not content or not chunk_size or not tokenizer:
        logger.debug(
            f"No chunking: content_empty={not content}, chunk_size={chunk_size}, tokenizer={tokenizer}")
        return [content] if content else []

    # Tokenize with offsets to preserve original text
    encoding = tokenizer.encode(content, add_special_tokens=False)
    token_ids = encoding.ids
    token_ids = [tid for tid in token_ids if tid != 0]  # Remove padding
    offsets = encoding.offsets

    header_encoding = tokenizer.encode(
        header_prefix, add_special_tokens=False) if header_prefix else None
    header_tokens = len(
        [tid for tid in header_encoding.ids if tid != 0]) if header_encoding else 0
    effective_chunk_size = max(1, chunk_size - buffer - header_tokens)
    logger.debug(
        f"Chunking with header_tokens={header_tokens}, effective_chunk_size={effective_chunk_size}")

    if len(token_ids) <= effective_chunk_size:
        logger.debug(f"Content fits in one chunk: {len(token_ids)} tokens")
        return [content]

    chunks = []
    start = 0
    step = effective_chunk_size - chunk_overlap
    while start < len(token_ids):
        end = min(start + effective_chunk_size, len(token_ids))
        # Extract substring using offsets
        start_offset = offsets[start][0] if start < len(
            offsets) else len(content)
        end_offset = offsets[end -
                             1][1] if end > 0 and end <= len(offsets) else len(content)
        chunk_text = content[start_offset:end_offset].strip()
        if chunk_text:
            chunks.append(chunk_text)
            logger.debug(
                f"Created chunk: tokens={end-start}, text_length={len(chunk_text)}")
        start += step
        if start >= len(token_ids) or (end == len(token_ids) and start + chunk_overlap >= len(token_ids)):
            break

    return chunks if chunks else [content]


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
    logger.debug(
        f"Processing node {node.id}: type={node.type}, header={node.header}, content_length={len(node.content)}, doc_id={node.doc_id}")
    result_nodes: List[TextNode] = []
    content = node.content.strip()

    # Handle empty content for TextNode
    if not content and isinstance(node, TextNode):
        logger.debug(f"Node {node.id} has empty content, returning unchanged")
        node.num_tokens = 0
        return [node]

    header_prefix = f"{node.header}\n" if node.header else ""

    # For HeaderNode, check if content needs chunking
    if isinstance(node, HeaderNode):
        full_content = f"{header_prefix}{content}" if content else header_prefix
        token_ids = tokenizer.encode(
            full_content, add_special_tokens=False).ids if tokenizer else []
        token_ids = [tid for tid in token_ids if tid != 0] if token_ids else []

        # Only chunk if content exceeds chunk_size
        if chunk_size and tokenizer and len(token_ids) > chunk_size - buffer:
            logger.debug(
                f"Chunking HeaderNode {node.id} with {len(token_ids)} tokens")
            chunks = chunk_content(
                content, tokenizer, chunk_size, chunk_overlap, buffer, header_prefix)
            for i, chunk in enumerate(chunks):
                full_content = f"{header_prefix}{chunk}" if chunk else header_prefix
                token_ids = tokenizer.encode(
                    full_content, add_special_tokens=False).ids
                token_ids = [tid for tid in token_ids if tid !=
                             0] if token_ids else []
                num_tokens = len(token_ids)
                if num_tokens > chunk_size - buffer:
                    logger.warning(
                        f"Chunk {i} for node {node.id} exceeds token limit: {num_tokens} > {chunk_size - buffer}")
                    continue
                new_node = create_text_node(
                    node, full_content, i, parent_id, parent_header)
                new_node.num_tokens = num_tokens
                logger.debug(
                    f"Created chunk {i} for node {node.id}: num_tokens={new_node.num_tokens}, doc_id={new_node.doc_id}")
                result_nodes.append(new_node)
        else:
            new_node = create_text_node(
                node, full_content, 0, parent_id, parent_header)
            token_ids = tokenizer.encode(
                full_content, add_special_tokens=False).ids if tokenizer else []
            token_ids = [tid for tid in token_ids if tid !=
                         0] if token_ids else []
            new_node.num_tokens = len(token_ids)
            logger.debug(
                f"Created single node {node.id}: num_tokens={new_node.num_tokens}, doc_id={new_node.doc_id}")
            result_nodes.append(new_node)

        # Process children
        for child in node.children:
            result_nodes.extend(
                process_node(child, tokenizer, chunk_size,
                             chunk_overlap, buffer, node.id, node.header)
            )
    else:
        # Handle TextNode with chunking
        chunks = chunk_content(
            content, tokenizer, chunk_size, chunk_overlap, buffer, header_prefix)
        for i, chunk in enumerate(chunks):
            full_content = f"{header_prefix}{chunk}" if chunk else header_prefix
            token_ids = tokenizer.encode(
                full_content, add_special_tokens=False).ids if tokenizer else []
            token_ids = [tid for tid in token_ids if tid !=
                         0] if token_ids else []
            num_tokens = len(token_ids)
            if chunk_size and num_tokens > chunk_size - buffer:
                logger.warning(
                    f"Chunk {i} for node {node.id} exceeds token limit: {num_tokens} > {chunk_size - buffer}")
                continue
            new_node = create_text_node(
                node, full_content, i, parent_id, parent_header)
            new_node.num_tokens = num_tokens
            logger.debug(
                f"Created chunk {i} for node {node.id}: num_tokens={new_node.num_tokens}, doc_id={new_node.doc_id}")
            result_nodes.append(new_node)

    return result_nodes


__all__ = [
    "process_node",
    "chunk_content",
    "create_text_node",
]
