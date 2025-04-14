from llama_index.core.schema import BaseNode, Document, NodeRelationship, NodeWithScore, RelatedNodeInfo, TextNode


def get_recursive_text(doc: Document, metadata_key: str) -> str:
    """
    Get content of this node and all of its child nodes recursively.
    """
    texts = [doc.metadata[metadata_key]]
    for child in doc.child_nodes or []:
        texts.append(child.metadata[metadata_key])
    return "\n".join(filter(None, texts))


def add_parent_child_relationship(parent_node: BaseNode, child_node: BaseNode) -> None:
    """Add parent/child relationship between nodes."""
    child_list: list[RelatedNodeInfo] = parent_node.child_nodes or []
    child_list.append(child_node.as_related_node_info())
    parent_node.relationships[NodeRelationship.CHILD] = child_list

    child_node.relationships[
        NodeRelationship.PARENT
    ] = parent_node.as_related_node_info()


def add_sibling_relationship(sibling_node1: BaseNode, sibling_node2: BaseNode) -> None:
    """Add parent/child relationship between nodes."""
    sibling_node1.relationships[NodeRelationship.NEXT] = sibling_node2.as_related_node_info(
    )
    sibling_node2.relationships[NodeRelationship.PREVIOUS] = sibling_node1.as_related_node_info(
    )
