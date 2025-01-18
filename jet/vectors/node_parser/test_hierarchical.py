import unittest
from jet.vectors.node_parser.hierarchical import JetHierarchicalNodeParser
from llama_index.core import Document
from llama_index.core.node_parser import (
    get_child_nodes,
    get_deeper_nodes,
    get_leaf_nodes,
    get_root_nodes,
)
from llama_index.core.node_parser.relational.hierarchical import HierarchicalNodeParser

ROOT_NODES_LEN = 1
CHILDREN_NODES_LEN = 3
GRAND_CHILDREN_NODES_LEN = 7


class TestNodeParser(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[512, 128, 64],
            chunk_overlap=10,
        )
        cls.nodes = node_parser.get_nodes_from_documents([Document.example()])

    def test_get_root_nodes(self):
        root_nodes = get_root_nodes(self.nodes)
        self.assertEqual(len(root_nodes), ROOT_NODES_LEN)

    def test_get_root_nodes_empty(self):
        root_nodes = get_root_nodes(get_leaf_nodes(self.nodes))
        self.assertEqual(root_nodes, [])

    def test_get_leaf_nodes(self):
        leaf_nodes = get_leaf_nodes(self.nodes)
        self.assertEqual(len(leaf_nodes), GRAND_CHILDREN_NODES_LEN)

    def test_get_child_nodes(self):
        child_nodes = get_child_nodes(
            get_root_nodes(self.nodes), all_nodes=self.nodes)
        self.assertEqual(len(child_nodes), CHILDREN_NODES_LEN)

    def test_get_deeper_nodes(self):
        deep_nodes = get_deeper_nodes(self.nodes, depth=0)
        self.assertEqual(deep_nodes, get_root_nodes(self.nodes))

        deep_nodes = get_deeper_nodes(self.nodes, depth=1)
        self.assertEqual(deep_nodes, get_child_nodes(
            get_root_nodes(self.nodes), self.nodes))

        deep_nodes = get_deeper_nodes(self.nodes, depth=2)
        self.assertEqual(deep_nodes, get_leaf_nodes(self.nodes))

        deep_nodes = get_deeper_nodes(self.nodes, depth=2)
        self.assertEqual(deep_nodes, get_child_nodes(get_child_nodes(
            get_root_nodes(self.nodes), self.nodes), self.nodes))

    def test_get_deeper_nodes_with_no_root_nodes(self):
        with self.assertRaises(ValueError, msg="There is no*"):
            get_deeper_nodes(get_leaf_nodes(self.nodes))

    def test_get_deeper_nodes_with_negative_depth(self):
        with self.assertRaises(ValueError, msg="Depth cannot be*"):
            get_deeper_nodes(self.nodes, -1)


if __name__ == "__main__":
    unittest.main()
