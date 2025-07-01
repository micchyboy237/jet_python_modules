from typing import Optional
from jet.data.header_utils import process_node
from jet.models.tokenizer.base import get_tokenizer, tokenize, detokenize
from jet.models.model_types import ModelType
from jet.data.header_types import HeaderNode, TextNode, NodeType, Nodes
from tokenizers import Tokenizer
from jet.logger import logger


def split_and_merge_headers(
    nodes: NodeType | Nodes,
    model: ModelType = "all-MiniLM-L6-v2",
    chunk_size: Optional[int] = None,
    chunk_overlap: int = 0,
    *,
    tokenizer: Optional[Tokenizer] = None,
    buffer: int = 0
) -> Nodes:
    """Split and merge headers in documents, handling both single nodes and lists."""
    if isinstance(nodes, (HeaderNode, TextNode)):
        nodes = [nodes]
    if not tokenizer and model:
        tokenizer = get_tokenizer(model)
    result_nodes: Nodes = []
    for node in nodes:
        logger.debug(
            f"Processing node {node.id}: header={node.header}, content_length={len(node.content)}")
        processed = process_node(
            node, tokenizer, chunk_size, chunk_overlap, buffer)
        result_nodes.extend(processed)
        logger.debug(f"Processed node {node.id}: got {len(processed)} nodes")
    logger.debug(f"Returning {len(result_nodes)} nodes after processing")
    return result_nodes
