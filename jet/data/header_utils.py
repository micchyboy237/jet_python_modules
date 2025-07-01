import uuid
from typing import List, Optional, Union, Literal
from pathlib import Path
from pydantic import BaseModel, Field
from jet.code.markdown_types import MarkdownToken, ContentType, MetaType
from jet.models.tokenizer.base import get_tokenizer, tokenize, detokenize
from jet.models.model_types import ModelType
from jet.data.header_types import Node, HeaderNode, TextNode, NodeType, Nodes
from tokenizers import Tokenizer
from jet.logger import logger


def split_and_merge_headers(
    docs: NodeType | Nodes,
    model: Optional[str | ModelType] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: int = 0,
    *,
    tokenizer: Optional[Tokenizer] = None,
    tokens: Optional[list[int] | list[list[int]]] = None,
    buffer: int = 0
) -> Nodes:
    """
    Split and merge nodes into chunks while preserving header hierarchy for context search and RAG.

    Args:
        docs: A single node or list of nodes to process.
        model: Model name or type for tokenization.
        chunk_size: Maximum token size for each chunk.
        chunk_overlap: Number of tokens to overlap between chunks.
        tokenizer: Optional pre-loaded tokenizer.
        tokens: Optional pre-computed tokens for the input.
        buffer: Additional token buffer for safety.

    Returns:
        List of nodes with split content, preserving header structure and chunk index.
    """
    if isinstance(docs, (HeaderNode, TextNode)):
        docs = [docs]

    if not tokenizer and model:
        tokenizer = get_tokenizer(model)

    result_nodes: Nodes = []

    def generate_unique_id() -> str:
        return f"auto_{uuid.uuid4().hex[:8]}"

    def process_node(node: NodeType, parent_id: Optional[str] = None, parent_header: Optional[str] = None) -> None:
        content = node.content.strip()
        if not content:
            result_nodes.append(node)
            return

        # Get header context
        current_header = node.header
        header_prefix = f"{current_header}\n" if current_header else ""

        # Tokenize content without special tokens for chunking
        token_ids = tokens if tokens else tokenize(
            content, tokenizer, add_special_tokens=False)
        if isinstance(token_ids[0], list):
            # Handle case where tokens is list[list[int]]
            token_ids = token_ids[0]

        effective_chunk_size = chunk_size - buffer if chunk_size else None
        logger.debug(
            f"Processing node with {len(token_ids)} tokens, effective_chunk_size={effective_chunk_size}, chunk_overlap={chunk_overlap}")

        if not effective_chunk_size or len(token_ids) <= effective_chunk_size:
            chunk_id = generate_unique_id()
            new_node = TextNode(
                id=chunk_id,
                line=node.line,
                parent_id=parent_id,
                parent_header=parent_header,
                type=node.type if isinstance(node, TextNode) else "paragraph",
                header=current_header,
                content=f"{header_prefix}{content}".strip(),
                meta=node.meta if isinstance(node, TextNode) else None,
                chunk_index=0
            )
            if parent_id:
                new_node._parent_node = node.get_parent_node()
            result_nodes.append(new_node)
        else:
            # Split tokens into chunks with overlap
            start = 0
            step = effective_chunk_size - chunk_overlap
            chunk_index = 0
            while start < len(token_ids):
                end = min(start + effective_chunk_size, len(token_ids))
                chunk_tokens = token_ids[start:end]
                logger.debug(
                    f"Creating chunk from tokens {start} to {end}, size={len(chunk_tokens)}, chunk_index={chunk_index}")
                chunk_text = detokenize(chunk_tokens, tokenizer)

                chunk_id = generate_unique_id()
                new_node = TextNode(
                    id=chunk_id,
                    line=node.line,
                    parent_id=parent_id,
                    parent_header=parent_header,
                    type=node.type if isinstance(
                        node, TextNode) else "paragraph",
                    header=current_header,
                    content=f"{header_prefix}{chunk_text}".strip(),
                    meta=node.meta if isinstance(node, TextNode) else None,
                    chunk_index=chunk_index
                )
                if parent_id:
                    new_node._parent_node = node.get_parent_node()
                result_nodes.append(new_node)

                chunk_index += 1
                start += step
                if start >= len(token_ids):
                    break
                # Ensure we don't create an empty chunk
                if end == len(token_ids) and start + chunk_overlap >= len(token_ids):
                    break

        # Process children for HeaderNode
        if isinstance(node, HeaderNode):
            for child in node.children:
                process_node(child, node.id, node.header)

    for node in docs:
        process_node(node)

    return result_nodes
