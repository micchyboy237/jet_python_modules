import pytest
from typing import List, Optional
from jet.code.markdown_types import ContentType
from jet.data.header_types import TextNode, HeaderNode
from jet.data.utils import generate_unique_id
from jet.data.header_utils import prepare_for_rag, VectorStore
from jet.models.tokenizer.base import get_tokenizer
from tokenizers import Tokenizer


@pytest.fixture
def tokenizer() -> Tokenizer:
    """Provide a tokenizer for tests."""
    return get_tokenizer("all-MiniLM-L6-v2")


@pytest.fixture
def default_params() -> dict:
    """Provide default parameters for chunking functions."""
    return {"chunk_size": 50, "chunk_overlap": 10, "buffer": 5}


class TestPrepareForRAG:
    @pytest.mark.parametrize(
        "node_content,meta,expected_content,expected_num_tokens",
        [
            ("Test Content", {}, "Test Content", lambda t: len([tid for tid in t.encode(
                "Test Header\nTest Content", add_special_tokens=False).ids if tid != 0])),
            ("", {}, "", lambda t: len([tid for tid in t.encode(
                "Test Header\n", add_special_tokens=False).ids if tid != 0])),
            ("Test Content", None, "Test Content", lambda t: len([tid for tid in t.encode(
                "Test Header\nTest Content", add_special_tokens=False).ids if tid != 0])),
        ],
        ids=["normal_content", "empty_content", "empty_meta"]
    )
    def test_prepare_single_node(self, tokenizer: Tokenizer, node_content: str, meta: Optional[dict], expected_content: str, expected_num_tokens) -> None:
        """Test preparing a single node for RAG with normal, empty content, and empty meta cases."""
        # Given
        expected_node = TextNode(
            id="node1",
            line=1,
            type="paragraph",
            header="Test Header",
            content=node_content,
            meta=meta,
            chunk_index=0,
            num_tokens=0
        )
        expected_embedding_shape = (384,)

        # When
        vector_store = prepare_for_rag(
            [expected_node], model="all-MiniLM-L6-v2", tokenizer=tokenizer)

        # Then
        result = vector_store.nodes
        assert len(result) == 1
        assert result[0].id == expected_node.id
        assert result[0].header == expected_node.header
        assert result[0].content == expected_content
        assert result[0].meta == meta
        assert result[0].num_tokens == expected_num_tokens(tokenizer)
        assert len(vector_store.embeddings) == 1
        assert vector_store.embeddings[0].shape == expected_embedding_shape

    def test_prepare_multiple_nodes_with_hierarchy(self, tokenizer: Tokenizer, default_params: dict) -> None:
        """Test preparing multiple nodes with and without hierarchy, including chunking."""
        # Given
        long_content = "This is a long sentence. " * 20
        expected_nodes = [
            TextNode(
                id="node1",
                line=1,
                type="paragraph",
                header="Header 1",
                content="Content 1",
                meta={},
                chunk_index=0,
                num_tokens=0
            ),
            HeaderNode(
                id="header2",
                line=2,
                type="header",
                header="Level 2 Header",
                content=long_content,
                level=1,
                children=[
                    TextNode(
                        id="child1",
                        line=3,
                        type="paragraph",
                        header="Child Header",
                        content="Child content",
                        meta={},
                        parent_id="header2",
                        parent_header="Level 2 Header",
                        chunk_index=0,
                        num_tokens=0
                    )
                ],
                chunk_index=0,
                num_tokens=0
            ),
            TextNode(
                id="node3",
                line=4,
                type="paragraph",
                header="Header 3",
                content="Content 3",
                meta={},
                parent_id="header2",
                parent_header="Level 2 Header",
                chunk_index=0,
                num_tokens=0
            )
        ]
        expected_embedding_shape = (384,)
        expected_min_chunks = 3

        # When
        vector_store = prepare_for_rag(
            expected_nodes,
            model="all-MiniLM-L6-v2",
            tokenizer=tokenizer,
            **default_params
        )

        # Then
        result = vector_store.nodes
        # At least 1 for node1, 3 for header2 chunks, 1 for node3
        assert len(result) >= expected_min_chunks + 2
        assert len(vector_store.embeddings) == len(result)
        for emb in vector_store.embeddings:
            assert emb.shape == expected_embedding_shape
        header_nodes = [n for n in result if n.header == "Level 2 Header"]
        assert len(header_nodes) >= expected_min_chunks
        for i, node in enumerate(header_nodes):
            assert node.content.startswith("Level 2 Header\n")
            assert node.num_tokens <= default_params["chunk_size"]
            assert node.num_tokens > 0
            assert node.chunk_index == i
        child_nodes = [
            n for n in result if n.parent_header == "Level 2 Header"]
        assert len(child_nodes) >= 1
        for node in child_nodes:
            if node.header == "Child Header":
                assert node.content == "Child Header\nChild content"
                assert node.parent_id == "header2"
            elif node.header == "Header 3":
                assert node.content == "Header 3\nContent 3"
                assert node.parent_id == "header2"

    def test_prepare_with_parent_header_deduplication(self, tokenizer: Tokenizer) -> None:
        """Test deduplication of parent_header when same as header."""
        # Given
        expected_node = TextNode(
            id="node1",
            line=1,
            type="paragraph",
            header="Test Header",
            content="Test Content",
            meta={},
            parent_header="Test Header",
            parent_id="parent1",
            chunk_index=0,
            num_tokens=0
        )
        expected_embedding_shape = (384,)
        expected_tokens = len([tid for tid in tokenizer.encode(
            "Test Header\nTest Content", add_special_tokens=False).ids if tid != 0])

        # When
        vector_store = prepare_for_rag(
            [expected_node], model="all-MiniLM-L6-v2", tokenizer=tokenizer)

        # Then
        result = vector_store.nodes
        assert len(result) == 1
        assert result[0].id == expected_node.id
        assert result[0].header == expected_node.header
        assert result[0].content == expected_node.content
        assert result[0].parent_header == expected_node.parent_header
        assert result[0].num_tokens == expected_tokens
        assert len(vector_store.embeddings) == 1
        assert vector_store.embeddings[0].shape == expected_embedding_shape

    def test_prepare_empty_nodes(self, tokenizer: Tokenizer) -> None:
        """Test preparing an empty node list."""
        # Given
        expected_nodes: List[TextNode] = []
        expected_embedding_shape = (0,)

        # When
        vector_store = prepare_for_rag(
            expected_nodes, model="all-MiniLM-L6-v2", tokenizer=tokenizer)

        # Then
        assert len(vector_store.nodes) == 0
        assert len(vector_store.embeddings) == 0
        assert vector_store.get_embeddings().shape == expected_embedding_shape

    def test_prepare_complex_hierarchy(self, tokenizer: Tokenizer, default_params: dict) -> None:
        """Test preparing a complex hierarchy with different node types and optional chunking."""
        # Given
        long_content = "This is a long sentence. " * 20
        expected_nodes = [
            HeaderNode(
                id="header1",
                line=1,
                type="header",
                header="Level 1 Header",
                content="Level 1 content",
                level=1,
                children=[
                    TextNode(
                        id="child1",
                        line=2,
                        type="code",
                        header="Code Child",
                        content=long_content,
                        meta={"language": "python"},
                        parent_id="header1",
                        parent_header="Level 1 Header",
                        chunk_index=0,
                        num_tokens=0
                    ),
                    TextNode(
                        id="child2",
                        line=3,
                        type="table",
                        header="Table Child",
                        content="| Col1 | Col2 |\n|------|------|\n| A    | B    |",
                        meta={"header": ["Col1", "Col2"],
                              "rows": [["A", "B"]]},
                        parent_id="header1",
                        parent_header="Level 1 Header",
                        chunk_index=0,
                        num_tokens=0
                    )
                ],
                chunk_index=0,
                num_tokens=0
            )
        ]
        expected_embedding_shape = (384,)
        expected_min_chunks = 3

        # When
        vector_store = prepare_for_rag(
            expected_nodes,
            model="all-MiniLM-L6-v2",
            tokenizer=tokenizer,
            **default_params
        )

        # Then
        result = vector_store.nodes
        # At least 1 for header, 3 for code child chunks, 1 for table
        assert len(result) >= expected_min_chunks + 2
        assert len(vector_store.embeddings) == len(result)
        for emb in vector_store.embeddings:
            assert emb.shape == expected_embedding_shape
        header_nodes = [n for n in result if n.header == "Level 1 Header"]
        assert len(header_nodes) == 1
        assert header_nodes[0].content == "Level 1 Header\nLevel 1 content"
        assert header_nodes[0].num_tokens > 0
        code_nodes = [n for n in result if n.header == "Code Child"]
        assert len(code_nodes) >= expected_min_chunks
        for i, node in enumerate(code_nodes):
            assert node.content.startswith("Code Child\n")
            assert node.num_tokens <= default_params["chunk_size"]
            assert node.num_tokens > 0
            assert node.chunk_index == i
            assert node.meta == {"language": "python"}
        table_nodes = [n for n in result if n.header == "Table Child"]
        assert len(table_nodes) == 1
        assert table_nodes[0].content == "Table Child\n| Col1 | Col2 |\n|------|------|\n| A    | B    |"
        assert table_nodes[0].meta == {"header": [
            "Col1", "Col2"], "rows": [["A", "B"]]}
