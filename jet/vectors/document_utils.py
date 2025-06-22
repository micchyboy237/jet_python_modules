from typing import List
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo

from jet.vectors.document_types import HeaderDocument


def get_leaf_documents(nodes: List[HeaderDocument]) -> List[HeaderDocument]:
    """
    Returns all leaf documents from a list of HeaderDocument nodes.
    A leaf document is a node with no child relationships.

    Args:
        nodes: List of HeaderDocument nodes to process.

    Returns:
        List of HeaderDocument nodes that have no child relationships.
    """
    leaf_nodes = []
    for node in nodes:
        child_rel = node.relationships.get(NodeRelationship.CHILD)
        if not child_rel:  # No child relationships means it's a leaf
            leaf_nodes.append(node)
        # If child_rel is a list of RelatedNodeInfo, it's not empty, so not a leaf
        # If child_rel is a single RelatedNodeInfo, it's not a leaf
    return leaf_nodes
