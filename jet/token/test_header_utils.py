import unittest
from typing import Callable, Union
from unittest.mock import patch, MagicMock
from jet.vectors.document_types import HeaderTextNode
from jet.utils.doc_utils import add_parent_child_relationship, add_sibling_relationship
from llama_index.core.schema import NodeRelationship
from jet.llm.embeddings.sentence_embedding import get_tokenizer_fn
# Assuming merge_headers is in a module named token_utils.py
from jet.token.token_utils import merge_headers


class TestMergeHeaders(unittest.TestCase):
    def setUp(self):
        # Mock tokenizer function for consistent token counts
        self.mock_tokenizer = MagicMock()
        self.model = "mistral"
        self.default_chunk_size = 100
        self.default_overlap = 20

        # Sample nodes for testing
        self.node1 = HeaderTextNode(
            text="This is a test header",
            metadata={"start_idx": 0, "end_idx": 20,
                      "chunk_index": None, "header_level": 1}
        )
        self.node2 = HeaderTextNode(
            text="Another header content",
            metadata={"start_idx": 21, "end_idx": 42,
                      "chunk_index": None, "header_level": 2}
        )
        self.node3 = HeaderTextNode(
            text="Short text",
            metadata={"start_idx": 43, "end_idx": 53,
                      "chunk_index": None, "header_level": 1}
        )

    def test_empty_input(self):
        """Test merge_headers with empty node list."""
        result = merge_headers(
            [], model=self.model, chunk_size=self.default_chunk_size, chunk_overlap=self.default_overlap)
        self.assertEqual(result, [], "Empty input should return empty list")

    def test_single_node_no_split(self):
        """Test merge_headers with a single node that doesn't need splitting."""
        def mock_tokenizer(texts: Union[str, list[str]]) -> list[int]:
            return [len(text.split()) for text in texts] if isinstance(texts, list) else len(texts.split())

        with patch("jet.token.token_utils.get_tokenizer_fn", return_value=mock_tokenizer):
            result = merge_headers(
                [self.node1],
                model=self.model,
                chunk_size=50,
                chunk_overlap=0
            )

        self.assertEqual(len(result), 1, "Should return one node")
        self.assertEqual(
            result[0].text, "This is a test header", "Node text should match input")
        self.assertEqual(result[0].metadata["chunk_index"],
                         0, "Chunk index should be set")
        self.assertEqual(result[0].metadata["start_idx"],
                         0, "Start index should match")
        self.assertEqual(result[0].metadata["end_idx"],
                         20, "End index should match")

    def test_merge_multiple_nodes(self):
        """Test merging multiple nodes into one chunk."""
        def mock_tokenizer(texts: Union[str, list[str]]) -> list[list[int]]:
            return [[1] * len(text.split()) for text in texts] if isinstance(texts, list) else [1] * len(texts.split())

        with patch("jet.token.token_utils.get_tokenizer_fn", return_value=mock_tokenizer):
            result = merge_headers(
                # "This is a test header" (5 tokens), "Short text" (2 tokens)
                [self.node1, self.node3],
                model=self.model,
                chunk_size=10,
                chunk_overlap=0
            )

        self.assertEqual(len(result), 1, "Should merge into one node")
        self.assertEqual(
            result[0].text, "This is a test header Short text", "Text should be concatenated")
        self.assertEqual(result[0].metadata["chunk_index"],
                         0, "Chunk index should be 0")
        self.assertEqual(result[0].metadata["start_idx"],
                         0, "Start index should be 0")
        self.assertEqual(result[0].metadata["end_idx"],
                         30, "End index should reflect total length")
        # Check relationships
        self.assertIn(NodeRelationship.CHILD,
                      result[0].relationships, "Merged node should have children")
        self.assertEqual(
            len(result[0].relationships[NodeRelationship.CHILD]),
            2,
            "Should have two child nodes"
        )

    def test_chunk_size_split(self):
        """Test splitting nodes due to chunk size limit."""
        def mock_tokenizer(texts: Union[str, list[str]]) -> list[list[int]]:
            return [[1] * len(text.split()) for text in texts] if isinstance(texts, list) else [1] * len(texts.split())

        with patch("jet.token.token_utils.get_tokenizer_fn", return_value=mock_tokenizer):
            result = merge_headers(
                [self.node1, self.node2],  # 5 tokens + 6 tokens
                model=self.model,
                chunk_size=6,
                chunk_overlap=0
            )

        self.assertEqual(len(result), 2, "Should split into two chunks")
        self.assertEqual(
            result[0].text, "This is a test header", "First chunk should contain node1")
        self.assertEqual(
            result[1].text, "Another header content", "Second chunk should contain node2")
        self.assertEqual(result[0].metadata["chunk_index"],
                         0, "First chunk index should be 0")
        self.assertEqual(result[1].metadata["chunk_index"],
                         1, "Second chunk index should be 1")

    def test_chunk_overlap(self):
        """Test handling of chunk overlap."""
        def mock_tokenizer(texts: Union[str, list[str]]) -> list[list[int]]:
            return [[1] * len(text.split()) for text in texts] if isinstance(texts, list) else [1] * len(texts.split())
        mock_tokenizer.decode = lambda tokens, **kwargs: " ".join(
            ["word"] * len(tokens))

        with patch("jet.token.token_utils.get_tokenizer_fn", return_value=mock_tokenizer):
            with patch("jet.token.token_utils.token_counter", side_effect=lambda text, model: len(text.split())):
                result = merge_headers(
                    [self.node1, self.node2],  # 5 tokens + 6 tokens
                    model=self.model,
                    chunk_size=6,
                    chunk_overlap=2
                )

        self.assertEqual(len(result), 2, "Should split into two chunks")
        self.assertTrue(
            "header" in result[1].text, "Second chunk should include overlap text")
        self.assertEqual(result[0].metadata["chunk_index"],
                         0, "First chunk index should be 0")
        self.assertEqual(result[1].metadata["chunk_index"],
                         1, "Second chunk index should be 1")

    def test_buffer_handling(self):
        """Test handling of buffer parameter."""
        def mock_tokenizer(texts: Union[str, list[str]]) -> list[list[int]]:
            return [[1] * len(text.split()) for text in texts] if isinstance(texts, list) else [1] * len(texts.split())

        with patch("jet.token.token_utils.get_tokenizer_fn", return_value=mock_tokenizer):
            result = merge_headers(
                [self.node1, self.node3],  # 5 tokens + 2 tokens
                model=self.model,
                chunk_size=8,
                chunk_overlap=0,
                buffer=2
            )

        self.assertEqual(
            len(result), 2, "Should split due to effective chunk size (6)")
        self.assertEqual(
            result[0].text, "This is a test header", "First chunk should contain node1")
        self.assertEqual(result[1].text, "Short text",
                         "Second chunk should contain node3")

    def test_invalid_chunk_size(self):
        """Test error handling for invalid chunk_size."""
        with self.assertRaises(ValueError, msg="Chunk size must be greater than chunk overlap"):
            merge_headers(
                [self.node1],
                model=self.model,
                chunk_size=10,
                chunk_overlap=10
            )

    def test_invalid_effective_max_tokens(self):
        """Test error handling for invalid effective max tokens."""
        with self.assertRaises(ValueError, msg="Effective max tokens must be greater than chunk overlap"):
            merge_headers(
                [self.node1],
                model=self.model,
                chunk_size=20,
                chunk_overlap=10,
                buffer=15
            )

    def test_relationships(self):
        """Test parent-child and sibling relationships."""
        def mock_tokenizer(texts: Union[str, list[str]]) -> list[list[int]]:
            return [[1] * len(text.split()) for text in texts] if isinstance(texts, list) else [1] * len(texts.split())

        with patch("jet.token.token_utils.get_tokenizer_fn", return_value=mock_tokenizer):
            result = merge_headers(
                [self.node1, self.node2, self.node3],
                model=self.model,
                chunk_size=20,
                chunk_overlap=0
            )

        self.assertEqual(len(result), 1, "Should merge into one chunk")
        merged_node = result[0]
        self.assertIn(NodeRelationship.CHILD, merged_node.relationships,
                      "Merged node should have children")
        children = merged_node.relationships[NodeRelationship.CHILD]
        self.assertEqual(len(children), 3, "Should have three child nodes")

        # Check sibling relationships
        self.assertIn(NodeRelationship.NEXT, self.node1.relationships,
                      "Node1 should have a next sibling")
        self.assertIn(NodeRelationship.PREVIOUS, self.node2.relationships,
                      "Node2 should have a previous sibling")
        self.assertEqual(
            self.node1.relationships[NodeRelationship.NEXT].node_id,
            self.node2.id_,
            "Node1's next sibling should be node2"
        )


if __name__ == "__main__":
    unittest.main()
