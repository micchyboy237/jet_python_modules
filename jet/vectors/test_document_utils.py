import pytest
from typing import List
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo

from jet.vectors.document_types import HeaderDocument
from jet.vectors.document_utils import get_leaf_documents


class TestGetLeafDocuments:
    def test_returns_empty_list_for_empty_input(self):
        # Given: An empty list of nodes
        nodes: List[HeaderDocument] = []

        # When: get_leaf_documents is called
        result = get_leaf_documents(nodes)

        # Then: An empty list is returned
        expected: List[HeaderDocument] = []
        assert result == expected

    def test_returns_leaf_nodes_with_no_children(self):
        # Given: A list with one node that has no children
        node1 = HeaderDocument(
            text="Leaf content",
            id="node1",
            metadata={"header": "Leaf Header"}
        )
        nodes = [node1]

        # When: get_leaf_documents is called
        result = get_leaf_documents(nodes)

        # Then: The node is returned as a leaf
        expected = [node1]
        assert result == expected

    def test_excludes_nodes_with_single_child(self):
        # Given: Two nodes, one with a child and one without
        node1 = HeaderDocument(
            text="Parent content",
            id="node1",
            metadata={"header": "Parent Header"},
            relationships={
                NodeRelationship.CHILD: RelatedNodeInfo(node_id="node2")
            }
        )
        node2 = HeaderDocument(
            text="Leaf content",
            id="node2",
            metadata={"header": "Leaf Header"}
        )
        nodes = [node1, node2]

        # When: get_leaf_documents is called
        result = get_leaf_documents(nodes)

        # Then: Only the node without children is returned
        expected = [node2]
        assert result == expected

    def test_excludes_nodes_with_multiple_children(self):
        # Given: Three nodes, one with multiple children and two leaves
        node1 = HeaderDocument(
            text="Parent content",
            id="node1",
            metadata={"header": "Parent Header"},
            relationships={
                NodeRelationship.CHILD: [
                    RelatedNodeInfo(node_id="node2"),
                    RelatedNodeInfo(node_id="node3")
                ]
            }
        )
        node2 = HeaderDocument(
            text="Leaf content 1",
            id="node2",
            metadata={"header": "Leaf Header 1"}
        )
        node3 = HeaderDocument(
            text="Leaf content 2",
            id="node3",
            metadata={"header": "Leaf Header 2"}
        )
        nodes = [node1, node2, node3]

        # When: get_leaf_documents is called
        result = get_leaf_documents(nodes)

        # Then: Only the nodes without children are returned
        expected = [node2, node3]
        assert result == expected
