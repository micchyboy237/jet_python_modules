from typing import Optional
from jet.data.header_utils import process_node
from jet.models.tokenizer.base import get_tokenizer, tokenize, detokenize
from jet.models.model_types import ModelType
from jet.data.header_types import HeaderNode, TextNode, NodeType, Nodes
from tokenizers import Tokenizer


def split_and_merge_headers(
    docs: NodeType | Nodes,
    model: ModelType = "all-MiniLM-L6-v2",
    chunk_size: Optional[int] = None,
    chunk_overlap: int = 0,
    *,
    tokenizer: Optional[Tokenizer] = None,
    tokens: Optional[list[int] | list[list[int]]] = None,
    buffer: int = 0
) -> Nodes:
    """Split and merge headers in documents, handling both single nodes and lists."""
    if isinstance(docs, (HeaderNode, TextNode)):
        docs = [docs]

    if not tokenizer and model:
        tokenizer = get_tokenizer(model)

    result_nodes: Nodes = []
    for node in docs:
        result_nodes.extend(
            process_node(node, tokenizer, chunk_size, chunk_overlap, buffer)
        )

    return result_nodes
