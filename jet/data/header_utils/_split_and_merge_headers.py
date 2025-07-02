from typing import Optional
from jet.data.header_utils import process_node
from jet.data.header_utils._base import process_nodes
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
    result_nodes = process_nodes(
        nodes, model, chunk_size, chunk_overlap, buffer)
    # result_nodes: Nodes = []
    # for node in nodes:
    #     logger.debug(
    #         f"Processing node {node.id}: header={node.header}, content_length={len(node.content)}, doc_id={node.doc_id}")
    #     processed = process_node(
    #         node, tokenizer, chunk_size, chunk_overlap, buffer)
    #     # Ensure doc_id is preserved in processed nodes
    #     for proc_node in processed:
    #         proc_node.doc_id = node.doc_id
    #     result_nodes.extend(processed)
    #     logger.debug(f"Processed node {node.id}: got {len(processed)} nodes")
    logger.debug(f"Returning {len(result_nodes)} nodes after processing")
    return result_nodes
