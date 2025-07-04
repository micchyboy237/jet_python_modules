from typing import List, Optional, Union, Literal
from pathlib import Path
from pydantic import BaseModel, Field
from jet.code.markdown_types import MarkdownToken, ContentType, MetaType
from jet.data.utils import generate_unique_id
from jet.models.tokenizer.base import get_tokenizer_fn, tokenize, detokenize
from jet.models.model_types import ModelType
from jet.data.header_types import Node, HeaderNode, TextNode, NodeType, Nodes
from tokenizers import Tokenizer
from jet.logger import logger


def create_text_node(
    node: NodeType,
    content: str,
    chunk_index: int,
    parent_id: Optional[str] = None,
    parent_header: Optional[str] = None,
    doc_id: Optional[str] = None
) -> TextNode:
    """Create a new TextNode with the given parameters."""
    new_node = TextNode(
        id=generate_unique_id(),
        doc_index=node.doc_index,
        line=node.line,
        parent_id=parent_id,
        parent_header=parent_header,
        type=node.type if isinstance(node, TextNode) else "paragraph",
        header=node.header,
        content=content.strip(),
        meta=node.meta if isinstance(node, TextNode) else None,
        chunk_index=chunk_index,
        doc_id=doc_id or node.doc_id
    )
    if parent_id:
        new_node._parent_node = node.get_parent_node()
    return new_node


def chunk_content(
    content: str,
    model_name_or_tokenizer: Optional[Union[ModelType, Tokenizer]],
    chunk_size: Optional[int],
    chunk_overlap: int,
    buffer: int,
    header_prefix: str = ""
) -> List[str]:
    """Split content into chunks based on token count, preserving original case."""
    if not content or not chunk_size or not model_name_or_tokenizer:
        logger.debug(
            f"No chunking: content_empty={not content}, chunk_size={chunk_size}, model_name_or_tokenizer={model_name_or_tokenizer}")
        return [content] if content else []

    tokenizer = get_tokenizer_fn(model_name_or_tokenizer)
    if not tokenizer:
        logger.debug(
            "No valid tokenizer provided, returning content as single chunk")
        return [content] if content else []

    encoding = tokenizer.encode(content, add_special_tokens=False)
    token_ids = [tid for tid in encoding if tid != 0]
    offsets = encoding.offsets
    header_encoding = tokenizer.encode(
        header_prefix, add_special_tokens=False) if header_prefix else None
    header_tokens = len(
        [tid for tid in header_encoding if tid != 0]) if header_encoding else 0
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
    model_name_or_tokenizer: Optional[Union[ModelType, Tokenizer]],
    chunk_size: Optional[int],
    chunk_overlap: int,
    buffer: int,
    max_length: Optional[int] = None,
    parent_id: Optional[str] = None,
    parent_header: Optional[str] = None
) -> List[TextNode]:
    """Process a single node and return a list of resulting TextNodes."""
    logger.debug(
        f"Processing node {node.id}: type={node.type}, header={node.header}, content_length={len(node.content)}, doc_id={node.doc_id}"
    )
    result_nodes: List[TextNode] = []
    node.calculate_num_tokens(model_name_or_tokenizer)
    content = node.content.strip()

    # Resolve tokenizer
    tokenizer = None
    if model_name_or_tokenizer:
        tokenizer = get_tokenizer_fn(model_name_or_tokenizer)

    # Handle empty content for TextNode
    if not content and isinstance(node, TextNode):
        logger.debug(f"Node {node.id} has empty content, returning unchanged")
        node.num_tokens = 0
        node.content = ""
        node.header = ""  # Ensure header is empty to match get_text()
        return [node]

    shared_doc_id = node.doc_id
    full_content = node.get_text()  # Use get_text() for full_content

    if isinstance(node, HeaderNode):
        # Calculate tokens for logging
        current_num_tokens = node.num_tokens
        logger.debug(
            f"Token count for node {node.id}: {current_num_tokens} tokens, get_text='{full_content}'")

        if chunk_size and tokenizer and current_num_tokens > chunk_size - buffer:
            logger.debug(
                f"Chunking HeaderNode {node.id} with {current_num_tokens} tokens"
            )
            chunks = chunk_content(
                content, model_name_or_tokenizer, chunk_size, chunk_overlap, buffer, node.header +
                "\n" if node.header else ""
            )
            for i, chunk in enumerate(chunks):
                chunk_text = f"{node.header}\n{chunk}" if node.header and chunk else chunk
                new_node = create_text_node(
                    node, chunk_text, i, parent_id, parent_header, doc_id=shared_doc_id
                )
                new_node.calculate_num_tokens(model_name_or_tokenizer)
                # Calculate num_tokens based on get_text()
                text_for_tokens = new_node.get_text()
                num_tokens = new_node.num_tokens
                logger.debug(
                    f"Chunk {i} token count: {num_tokens}, get_text='{text_for_tokens}'")
                if max_length and num_tokens > max_length - buffer:
                    logger.warning(
                        f"Chunk {i} for node {node.id} exceeds token limit: {num_tokens} > {chunk_size - buffer}"
                    )
                    continue

                logger.debug(
                    f"Created chunk {i} for node {node.id}: num_tokens={new_node.num_tokens}, doc_id={new_node.doc_id}, type={new_node.type}"
                )
                result_nodes.append(new_node)
        else:
            new_node = create_text_node(
                node, full_content, 0, parent_id, parent_header, doc_id=shared_doc_id
            )
            new_node.calculate_num_tokens(model_name_or_tokenizer)
            # Calculate num_tokens based on get_text()
            text_for_tokens = new_node.get_text()
            logger.debug(
                f"Created single node {node.id}: num_tokens={new_node.num_tokens}, doc_id={new_node.doc_id}, type={new_node.type}, get_text='{text_for_tokens}'"
            )
            result_nodes.append(new_node)
        # for child in node.children:
        #     result_nodes.extend(
        #         process_node(child, model_name_or_tokenizer, chunk_size,
        #                      chunk_overlap, buffer, max_length, parent_id=node.id, parent_header=node.header)
        #     )
    else:
        chunks = chunk_content(
            content, model_name_or_tokenizer, chunk_size, chunk_overlap, buffer, node.header +
            "\n" if node.header else ""
        )
        for i, chunk in enumerate(chunks):
            chunk_text = f"{node.header}\n{chunk}" if node.header and chunk else chunk
            new_node = create_text_node(
                node, chunk_text, i, parent_id, parent_header, doc_id=shared_doc_id
            )
            new_node.calculate_num_tokens(model_name_or_tokenizer)
            # Calculate num_tokens based on get_text()
            text_for_tokens = new_node.get_text()
            num_tokens = new_node.num_tokens
            logger.debug(
                f"TextNode chunk {i} token count: {num_tokens}, get_text='{text_for_tokens}'")
            if chunk_size and num_tokens > chunk_size - buffer:
                logger.warning(
                    f"Chunk {i} for node {node.id} exceeds token limit: {num_tokens} > {chunk_size - buffer}"
                )
                continue

            logger.debug(
                f"Created chunk {i} for node {node.id}: num_tokens={new_node.num_tokens}, doc_id={new_node.doc_id}, type={new_node.type}"
            )
            result_nodes.append(new_node)

    return result_nodes


def process_nodes(
    nodes: List[NodeType],
    model_name_or_tokenizer: Optional[Union[ModelType, Tokenizer]],
    chunk_size: Optional[int],
    chunk_overlap: int,
    buffer: int,
    max_length: Optional[int] = None,
    parent_id: Optional[str] = None,
    parent_header: Optional[str] = None,
    processed_ids: Optional[set[str]] = None
) -> List[TextNode]:
    """Process a list of nodes, applying chunking and maintaining parent relationships, avoiding duplicate processing."""
    if processed_ids is None:
        processed_ids = set()

    result_nodes: List[TextNode] = []
    for node in nodes:
        if node.id in processed_ids:
            logger.warning(f"Skipping duplicate node {node.id}")
            continue

        processed_ids.add(node.id)

        # Process the current node with the provided parent_id and parent_header
        processed_nodes = process_node(
            node=node,
            model_name_or_tokenizer=model_name_or_tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            buffer=buffer,
            max_length=max_length,
            parent_id=parent_id,
            parent_header=parent_header
        )
        result_nodes.extend(processed_nodes)

        # Process child nodes if any, passing the current HeaderNode's id and header as parent context
        if isinstance(node, HeaderNode) and hasattr(node, 'children') and node.children:
            child_nodes = process_nodes(
                nodes=node.children,
                model_name_or_tokenizer=model_name_or_tokenizer,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                buffer=buffer,
                max_length=max_length,
                parent_id=node.id,
                parent_header=node.header,
                processed_ids=processed_ids
            )
            result_nodes.extend(child_nodes)

    return result_nodes


def merge_nodes(
    nodes: Nodes,
    tokenizer: Tokenizer,
    max_tokens: int,
    buffer: int = 0
) -> List[TextNode]:
    """Merge nodes hierarchically while respecting max token limits."""
    logger.debug(
        f"Starting merge_nodes with {len(nodes)} nodes, max_tokens={max_tokens}, buffer={buffer}")
    if not nodes:
        logger.debug("No nodes provided, returning empty list")
        return []

    result_nodes: List[TextNode] = []
    current_group: List[NodeType] = []
    current_token_count: int = 0
    current_chunk_index: int = 0

    def create_merged_node(group: List[NodeType], chunk_index: int) -> TextNode:
        logger.debug(
            f"Creating merged node for group of {len(group)} nodes, chunk_index={chunk_index}")
        texts = []
        for node in group:
            if node.get_text().strip():
                texts.append(node.get_text())
                logger.debug(f"Adding text: {node.get_text()}")
        combined_text = "\n".join(texts)

        parent_headers = list(
            set(node.parent_header for node in group if node.parent_header))
        parent_header = parent_headers[0] if parent_headers else None
        doc_ids = list(set(node.doc_id for node in group))
        doc_id = doc_ids[0] if doc_ids else group[0].doc_id
        parent_ids = list(
            set(node.parent_id for node in group if node.parent_id))
        parent_id = parent_ids[0] if parent_ids else None

        new_node = create_text_node(
            node=group[0],
            content=combined_text,
            chunk_index=chunk_index,
            parent_id=parent_id,
            parent_header=parent_header,
            doc_id=doc_id
        )
        token_ids = tokenizer.encode(
            combined_text, add_special_tokens=False).ids
        new_node.num_tokens = len(token_ids)
        logger.debug(
            f"Merged node created: text='{combined_text}', num_tokens={new_node.num_tokens}")
        return new_node

    for i, node in enumerate(nodes):
        if not node.get_text().strip():
            logger.debug(f"Skipping empty node {node.id}")
            continue

        token_ids = tokenizer.encode(
            node.get_text(), add_special_tokens=False).ids
        token_count = len(token_ids)
        logger.debug(
            f"Node {node.id}: text='{node.get_text()}', token_count={token_count}")

        if token_count > max_tokens - buffer:
            logger.debug(
                f"Node {node.id} exceeds token limit: {token_count} > {max_tokens - buffer}")
            chunks = chunk_content(
                content=node.content,
                model_name_or_tokenizer=tokenizer,
                chunk_size=max_tokens,
                chunk_overlap=0,
                buffer=buffer,
                header_prefix=node.header + "\n" if node.header else ""
            )
            for j, chunk in enumerate(chunks):
                chunk_text = f"{node.header}\n{chunk}" if node.header and chunk else chunk
                new_node = create_text_node(
                    node=node,
                    content=chunk_text,
                    chunk_index=current_chunk_index,
                    parent_id=node.parent_id,
                    parent_header=node.parent_header,
                    doc_id=node.doc_id
                )
                token_ids = tokenizer.encode(
                    chunk_text, add_special_tokens=False).ids
                new_node.num_tokens = len(token_ids)
                logger.debug(
                    f"Chunk {j} created: text='{chunk_text}', num_tokens={new_node.num_tokens}")
                result_nodes.append(new_node)
                current_chunk_index += 1
            continue

        if current_token_count + token_count <= max_tokens - buffer:
            current_group.append(node)
            current_token_count += token_count
            logger.debug(
                f"Added node {node.id} to current group, total_tokens={current_token_count}")
        else:
            if current_group:
                result_nodes.append(create_merged_node(
                    current_group, current_chunk_index))
                current_chunk_index += 1
                logger.debug(
                    f"Flushed group, new chunk_index={current_chunk_index}")
            current_group = [node]
            current_token_count = token_count
            logger.debug(
                f"Started new group with node {node.id}, token_count={token_count}")

    if current_group:
        result_nodes.append(create_merged_node(
            current_group, current_chunk_index))
        logger.debug("Flushed final group")

    logger.debug(f"Merge completed, returning {len(result_nodes)} nodes")
    return result_nodes


__all__ = [
    "create_text_node",
    "chunk_content",
    "process_node",
    "process_nodes",
    "merge_nodes",
]
