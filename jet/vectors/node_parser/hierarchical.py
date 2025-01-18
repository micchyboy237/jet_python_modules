from typing import Any, Dict, List, Optional, Sequence
from llama_index.core.schema import BaseNode, Document, NodeRelationship, RelatedNodeInfo
from llama_index.core.node_parser.relational.hierarchical import HierarchicalNodeParser, get_deeper_nodes, get_root_nodes


class JetHierarchicalNodeParser:
    """Jet HierarchicalNodeParser with depth calculation functionality."""

    all_nodes: list[BaseNode] = []
    depth: int = 0  # Variable to store the calculated depth
    nodes_tree: dict = {}  # Variable to store the calculated depth
    chunk_sizes: Optional[List[int]] = None

    def __init__(
        self,
        all_nodes: list[BaseNode],
        chunk_sizes: List[int],
    ):
        self.all_nodes = all_nodes
        self.chunk_sizes = chunk_sizes
        self.depth = self.get_depth(all_nodes)
        self.nodes_tree = self.generate_nodes_tree(all_nodes)

    def get_depth(self, nodes: List[BaseNode]) -> int:
        """Get the maximum depth of the hierarchical node structure."""
        def _get_node_depth(node: BaseNode, all_nodes: List[BaseNode], current_depth: int = 1) -> int:
            """Recursively get the depth of a node."""
            # If the node has no children, it is a leaf node, so return the current depth.
            if not hasattr(node, "relationships") or NodeRelationship.CHILD not in node.relationships:
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

    def generate_nodes_tree(self, nodes: List[BaseNode], depth: Optional[int] = None):
        max_depth = depth if isinstance(depth, int) else self.depth
        if not isinstance(max_depth, int):
            raise ValueError(f"Invalide max_depth: {max_depth}")
        if not self.chunk_sizes:
            raise ValueError(f"Invalide chunk_sizes: {self.chunk_sizes}")

        for idx, chunk_size in enumerate(self.chunk_sizes):
            if idx <= max_depth:
                deep_nodes = get_deeper_nodes(nodes, depth=idx)
                for node in deep_nodes:
                    node.metadata["chunk_size"] = chunk_size
                    node.metadata["depth"] = idx

        self.nodes_tree = self._generate_nodes_tree()
        return self.nodes_tree

    def _generate_nodes_tree(self, nodes: Optional[list[BaseNode] | list[RelatedNodeInfo]] = None, depth=0, nodes_tree: dict = {}):
        nodes = get_deeper_nodes(self.all_nodes, depth=depth)
        for node in nodes:
            metadata = node.metadata.copy()
            sub_nodes_tree = {}
            nodes_tree[node.node_id] = {
                "node_id": node.node_id,
                "text": node.text,
                "chunk_size": metadata.pop("chunk_size"),
                "depth": metadata.pop("depth"),
                "metadata": metadata,
                "sub_nodes": sub_nodes_tree
            }
            child_nodes = node.child_nodes
            self._generate_nodes_tree(
                child_nodes, depth=depth+1, nodes_tree=sub_nodes_tree)
        return nodes_tree

    def get_nodes_from_documents(
        self,
        documents: Sequence[Document],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        """Parse document into nodes and calculate the depth."""
        all_nodes = self.node_parser.get_nodes_from_documents(
            documents, show_progress, **kwargs)
        self.all_nodes = all_nodes

        # After parsing nodes, calculate and store the depth
        # This will store the depth in self.depth
        depth = self.get_depth(all_nodes)
        nodes_tree = self.generate_nodes_tree(all_nodes, depth)
        return {
            "all_nodes": all_nodes,
            "nodes_tree": nodes_tree,
            "depth": depth,
        }

    @property
    def readable_depth(self) -> Optional[int]:
        """Read-only property to access the depth."""
        return self.depth

    @property
    def readable_nodes_tree(self):
        pass


__all__ = [
    "JetHierarchicalNodeParser"
]
