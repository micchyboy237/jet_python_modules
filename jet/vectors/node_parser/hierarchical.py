from typing import Any, Dict, List, Optional, Sequence
from llama_index.core.schema import BaseNode, Document, NodeRelationship
from llama_index.core.node_parser.relational.hierarchical import HierarchicalNodeParser, get_root_nodes


class JetHierarchicalNodeParser(HierarchicalNodeParser):
    """Jet HierarchicalNodeParser with depth calculation functionality."""

    depth: Optional[int] = None  # Variable to store the calculated depth

    def get_depth(self, nodes: List[BaseNode]) -> int:
        """Get the maximum depth of the hierarchical node structure."""
        def _get_node_depth(node: BaseNode, all_nodes: List[BaseNode], current_depth: int = 1) -> int:
            """Recursively get the depth of a node."""
            # If the node has no children, it is a leaf node, so return the current depth.
            if NodeRelationship.CHILD not in node.relationships:
                return current_depth

            # If the node has children, recursively calculate the depth of each child.
            max_child_depth = current_depth
            for child_node in node.child_nodes or []:
                child_depth = _get_node_depth(
                    child_node, all_nodes, current_depth + 1)
                max_child_depth = max(max_child_depth, child_depth)

            return max_child_depth

        # Find root nodes, since depth is calculated from them
        root_nodes = get_root_nodes(nodes)
        if not root_nodes:
            raise ValueError("No root nodes found.")

        # Calculate the depth for each root node and store it in the class-level depth variable
        self.depth = max(_get_node_depth(root, nodes) for root in root_nodes)
        return self.depth

    def get_nodes_from_documents(
        self,
        documents: Sequence[Document],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        """Parse document into nodes and calculate the depth."""
        all_nodes = super().get_nodes_from_documents(documents, show_progress, **kwargs)

        # After parsing nodes, calculate and store the depth
        self.get_depth(all_nodes)  # This will store the depth in self.depth
        return all_nodes

    @property
    def readable_depth(self) -> Optional[int]:
        """Read-only property to access the depth."""
        return self.depth


__all__ = [
    "JetHierarchicalNodeParser"
]
