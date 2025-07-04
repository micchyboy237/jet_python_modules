import pytest
from typing import List, Optional
from tokenizers import Tokenizer
from jet.data.header_types import NodeType, TextNode
from jet.data.header_utils._base import create_text_node, merge_nodes
from jet.mocks.mock_tokenizer import MockTokenizer
from jet.utils.text_constants import TEXT_CONTRACTIONS_EN
from jet.logger import logger


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()


@pytest.fixture
def sample_nodes():
    nodes = [
        TextNode(
            id="node1",
            doc_index=0,
            line=1,
            type="paragraph",
            header="Header 1",
            content="This is a short sentence.",
            meta={},
            parent_id=None,
            parent_header=None,
            chunk_index=0,
            num_tokens=5,  # "This is a short sentence."
            doc_id="doc1"
        ),
        TextNode(
            id="node2",
            doc_index=1,
            line=2,
            type="paragraph",
            header="Header 2",
            content="This is another sentence with more words to test token limits.",
            meta={},
            parent_id="node1",
            parent_header="Header 1",
            chunk_index=0,
            num_tokens=10,  # "This is another sentence with more words to test token limits."
            doc_id="doc1"
        ),
        TextNode(
            id="node3",
            doc_index=2,
            line=3,
            type="paragraph",
            header="",
            content="",
            meta={},
            parent_id=None,
            parent_header=None,
            chunk_index=0,
            num_tokens=0,
            doc_id="doc1"
        ),
    ]
    logger.debug(
        f"Sample nodes created: {[node.get_text() for node in nodes]}")
    return nodes


class TestMergeNodes:
    def test_merge_nodes_within_token_limit(self, mock_tokenizer, sample_nodes):
        # Given: A list of nodes with combined token count under max_tokens
        nodes = sample_nodes[:2]
        max_tokens = 20
        expected_text = "Header 1\nThis is a short sentence.\nHeader 2\nThis is another sentence with more words to test token limits."
        expected_token_count = 15
        expected_parent_header = "Header 1"
        expected_doc_id = "doc1"
        expected_chunk_index = 0

        # When: Merging nodes with a mock tokenizer
        logger.debug(
            f"Merging nodes: {[node.get_text() for node in nodes]} with max_tokens={max_tokens}")
        result = merge_nodes(nodes, mock_tokenizer, max_tokens)
        logger.debug(f"Merge result: {[node.get_text() for node in result]}")

        # Then: Nodes are merged into one with correct metadata and token count
        assert len(result) == 1, f"Expected 1 node, got {len(result)}"
        assert result[0].get_text(
        ) == expected_text, f"Expected text: {expected_text}, got: {result[0].get_text()}"
        assert result[0].num_tokens == expected_token_count, f"Expected token count: {expected_token_count}, got: {result[0].num_tokens}"
        assert result[
            0].parent_header == expected_parent_header, f"Expected parent_header: {expected_parent_header}, got: {result[0].parent_header}"
        assert result[0].doc_id == expected_doc_id, f"Expected doc_id: {expected_doc_id}, got: {result[0].doc_id}"
        assert result[0].chunk_index == expected_chunk_index, f"Expected chunk_index: {expected_chunk_index}, got: {result[0].chunk_index}"

    def test_merge_nodes_exceeding_token_limit(self, mock_tokenizer, sample_nodes):
        # Given: A list of nodes where one exceeds max_tokens
        large_node = TextNode(
            id="large_node",
            doc_index=0,
            line=1,
            type="paragraph",
            header="Large Header",
            content=" ".join(["word"] * 30),  # 30 tokens
            meta={},
            parent_id=None,
            parent_header=None,
            chunk_index=0,
            num_tokens=30,
            doc_id="doc2"
        )
        nodes = [large_node]
        max_tokens = 15
        expected = [
            {
                "header": "Large Header - Part 1",
                "content": "Large Header - Part 1\n" + " ".join(["word"] * 14),
                "num_tokens": pytest.approx(15, abs=1),
                "parent_header": None,
                "doc_id": "doc2",
                "chunk_index": 0
            },
            {
                "header": "Large Header - Part 2",
                "content": "Large Header - Part 2\n" + " ".join(["word"] * 14),
                "num_tokens": pytest.approx(15, abs=1),
                "parent_header": None,
                "doc_id": "doc2",
                "chunk_index": 1
            },
        ]

        # When: Merging nodes with a mock tokenizer
        logger.debug(
            f"Merging large node: {large_node.get_text()} with max_tokens={max_tokens}")
        result = merge_nodes(nodes, mock_tokenizer, max_tokens)
        logger.debug(f"Merge result: {[node.get_text() for node in result]}")

        # Then: Large node is split into chunks with correct headers and metadata
        assert len(result) == 2, f"Expected 2 nodes, got {len(result)}"
        for i, res in enumerate(result):
            assert res.get_text(
            ) == expected[i]["content"], f"Chunk {i} text mismatch: expected {expected[i]['content']}, got {res.get_text()}"
            assert abs(res.num_tokens - expected[i]["num_tokens"]
                       ) <= 1, f"Chunk {i} token count mismatch: expected {expected[i]['num_tokens']}, got {res.num_tokens}"
            assert res.parent_header == expected[i][
                "parent_header"], f"Chunk {i} parent_header mismatch: expected {expected[i]['parent_header']}, got {res.parent_header}"
            assert res.doc_id == expected[i][
                "doc_id"], f"Chunk {i} doc_id mismatch: expected {expected[i]['doc_id']}, got {res.doc_id}"
            assert res.chunk_index == expected[i][
                "chunk_index"], f"Chunk {i} chunk_index mismatch: expected {expected[i]['chunk_index']}, got {res.chunk_index}"

    def test_merge_nodes_with_empty_node(self, mock_tokenizer, sample_nodes):
        # Given: A list of nodes including an empty node
        nodes = sample_nodes
        max_tokens = 20
        expected_text = "Header 1\nThis is a short sentence.\nThis is another sentence with more words to test token limits."
        expected_token_count = 15
        expected_parent_header = "Header 1"
        expected_doc_id = "doc1"
        expected_chunk_index = 0

        # When: Merging nodes with a mock tokenizer
        logger.debug(
            f"Merging nodes with empty: {[node.get_text() for node in nodes]} with max_tokens={max_tokens}")
        result = merge_nodes(nodes, mock_tokenizer, max_tokens)
        logger.debug(f"Merge result: {[node.get_text() for node in result]}")

        # Then: Empty node is skipped, others are merged correctly
        assert len(result) == 1, f"Expected 1 node, got {len(result)}"
        assert result[0].get_text(
        ) == expected_text, f"Expected text: {expected_text}, got: {result[0].get_text()}"
        assert result[0].num_tokens == expected_token_count, f"Expected token count: {expected_token_count}, got: {result[0].num_tokens}"
        assert result[
            0].parent_header == expected_parent_header, f"Expected parent_header: {expected_parent_header}, got: {result[0].parent_header}"
        assert result[0].doc_id == expected_doc_id, f"Expected doc_id: {expected_doc_id}, got: {result[0].doc_id}"
        assert result[0].chunk_index == expected_chunk_index, f"Expected chunk_index: {expected_chunk_index}, got: {result[0].chunk_index}"

    def test_merge_nodes_with_buffer(self, mock_tokenizer, sample_nodes):
        # Given: A list of nodes with a buffer reducing effective max_tokens
        nodes = sample_nodes[:2]
        max_tokens = 20
        buffer = 5
        expected = [
            {
                "header": "Header 1",
                "content": "Header 1\nThis is a short sentence.",
                "num_tokens": 5,
                "parent_header": "Header 1",
                "doc_id": "doc1",
                "chunk_index": 0
            },
            {
                "header": "Header 2",
                "content": "Header 2\nThis is another sentence with more words to test token limits.",
                "num_tokens": 10,
                "parent_header": "Header 1",
                "doc_id": "doc1",
                "chunk_index": 1
            }
        ]

        # When: Merging nodes with a buffer
        logger.debug(
            f"Merging nodes with buffer: {[node.get_text() for node in nodes]} with max_tokens={max_tokens}, buffer={buffer}")
        result = merge_nodes(nodes, mock_tokenizer, max_tokens, buffer=buffer)
        logger.debug(f"Merge result: {[node.get_text() for node in result]}")

        # Then: Nodes are split due to buffer reducing effective token limit
        assert len(result) == 2, f"Expected 2 nodes, got {len(result)}"
        for i, res in enumerate(result):
            assert res.get_text(
            ) == expected[i]["content"], f"Chunk {i} text mismatch: expected {expected[i]['content']}, got {res.get_text()}"
            assert res.num_tokens == expected[i][
                "num_tokens"], f"Chunk {i} token count mismatch: expected {expected[i]['num_tokens']}, got {res.num_tokens}"
            assert res.parent_header == expected[i][
                "parent_header"], f"Chunk {i} parent_header mismatch: expected {expected[i]['parent_header']}, got {res.parent_header}"
            assert res.doc_id == expected[i][
                "doc_id"], f"Chunk {i} doc_id mismatch: expected {expected[i]['doc_id']}, got {res.doc_id}"
            assert res.chunk_index == expected[i][
                "chunk_index"], f"Chunk {i} chunk_index mismatch: expected {expected[i]['chunk_index']}, got {res.chunk_index}"

    def test_merge_nodes_no_nodes(self, mock_tokenizer):
        # Given: An empty list of nodes
        nodes = []
        max_tokens = 20
        expected: List[TextNode] = []

        # When: Merging empty nodes
        logger.debug(f"Merging empty nodes with max_tokens={max_tokens}")
        result = merge_nodes(nodes, mock_tokenizer, max_tokens)
        logger.debug(f"Merge result: {result}")

        # Then: Returns empty list
        assert result == expected, f"Expected empty list, got {result}"

    def test_merge_nodes_with_special_characters(self, mock_tokenizer):
        # Given: A node with special characters and contractions
        nodes = [
            TextNode(
                id="node1",
                doc_index=0,
                line=1,
                type="paragraph",
                header="Special Header",
                content="It's a test with special chars: @#$% and contractions.",
                meta={},
                parent_id=None,
                parent_header=None,
                chunk_index=0,
                num_tokens=10,
                doc_id="doc3"
            )
        ]
        max_tokens = 20
        expected_text = "Special Header\nIt's a test with special chars: @#$% and contractions."
        expected_token_count = 10
        expected_parent_header = None
        expected_doc_id = "doc3"
        expected_chunk_index = 0

        # When: Merging node with special characters
        logger.debug(
            f"Merging node with special chars: {nodes[0].get_text()} with max_tokens={max_tokens}")
        result = merge_nodes(nodes, mock_tokenizer, max_tokens)
        logger.debug(f"Merge result: {[node.get_text() for node in result]}")

        # Then: Node is preserved with correct content and metadata
        assert len(result) == 1, f"Expected 1 node, got {len(result)}"
        assert result[0].get_text(
        ) == expected_text, f"Expected text: {expected_text}, got: {result[0].get_text()}"
        assert result[0].num_tokens == expected_token_count, f"Expected token count: {expected_token_count}, got: {result[0].num_tokens}"
        assert result[
            0].parent_header == expected_parent_header, f"Expected parent_header: {expected_parent_header}, got: {result[0].parent_header}"
        assert result[0].doc_id == expected_doc_id, f"Expected doc_id: {expected_doc_id}, got: {result[0].doc_id}"
        assert result[0].chunk_index == expected_chunk_index, f"Expected chunk_index: {expected_chunk_index}, got: {result[0].chunk_index}"
