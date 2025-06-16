import pytest
from typing import Any, Dict, List
from jet.vectors.document_types import Document, HeaderDocument, HeaderTextNode, HeaderDocumentWithScore, HeaderMetadata, Match
from llama_index.core.schema import MetadataMode, NodeRelationship, RelatedNodeInfo


class TestDocument:
    def test_init(self):
        expected_text = "Sample document"
        expected_metadata = {"key": "value"}
        doc = Document(text=expected_text, metadata=expected_metadata)
        result_text = doc.text
        result_metadata = doc.metadata
        assert result_text == expected_text
        assert result_metadata == expected_metadata

    def test_get_recursive_text(self):
        doc = Document(text="Main content")
        child1 = Document(text="Child1", metadata={"header": "Header1"})
        child2 = Document(text="Child2", metadata={"header": "Header2"})
        parent = Document(text="Parent", metadata={"header": "ParentHeader"})

        # Set relationships using RelatedNodeInfo
        doc.relationships = {
            NodeRelationship.CHILD: [
                RelatedNodeInfo(node_id=child1.node_id),
                RelatedNodeInfo(node_id=child2.node_id)
            ],
            NodeRelationship.PARENT: RelatedNodeInfo(node_id=parent.node_id)
        }
        child1.relationships = {
            NodeRelationship.PARENT: RelatedNodeInfo(node_id=doc.node_id)}
        child2.relationships = {
            NodeRelationship.PARENT: RelatedNodeInfo(node_id=doc.node_id)}
        parent.relationships = {NodeRelationship.CHILD: [
            RelatedNodeInfo(node_id=doc.node_id)]}

        # Create a node registry for testing
        node_registry = {
            doc.node_id: doc,
            child1.node_id: child1,
            child2.node_id: child2,
            parent.node_id: parent
        }

        expected = "Main content\n\nParentHeader\nHeader1\nHeader2"
        result = doc.get_recursive_text(node_registry=node_registry)
        assert result == expected


class TestHeaderDocument:
    def test_init_with_metadata(self):
        expected_id = "doc1"
        expected_text = "Sample content"
        expected_metadata: HeaderMetadata = {
            "id": expected_id,
            "doc_index": 0,
            "header_level": 1,
            "header": "Introduction",
            "parent_header": "Chapter 1",
            "content": expected_text,
            "chunk_index": None,
            "tokens": None,
            "source_url": None,
            "links": None,
            "texts": [expected_text],
        }
        doc = HeaderDocument(
            id=expected_id, text=expected_text, metadata=expected_metadata)
        result_id = doc.id
        result_text = doc.text
        result_metadata = doc.metadata
        assert result_id == expected_id
        assert result_text == expected_text
        assert result_metadata == expected_metadata
        assert doc.node_id == expected_id

    def test_getitem_metadata_access(self):
        doc = HeaderDocument(id="doc1", text="Content", metadata={
                             "header": "Intro", "header_level": 1})
        expected_header = "Intro"
        expected_level = 1
        result_header = doc["header"]
        result_level = doc["header_level"]
        assert result_header == expected_header
        assert result_level == expected_level

    def test_iter(self):
        doc = HeaderDocument(id="doc1", text="Content", metadata={
                             "header": "Intro", "header_level": 1})
        expected_items = {
            "text": "Content",
            "metadata": doc.metadata,
            "metadata_separator": doc.metadata_separator,
            "id": "doc1",
            "header": "Intro",
            "header_level": 1,
            "doc_index": 0,
            "parent_header": "",
            "content": "",
            "texts": ["Content"],
        }
        result = dict(doc)
        for key, expected_value in expected_items.items():
            assert key in result
            assert result[key] == expected_value

    def test_get(self):
        doc = HeaderDocument(id="doc1", text="Content",
                             metadata={"header": "Intro"})
        expected_header = "Intro"
        expected_default = "default"
        result_header = doc.get("header")
        result_nonexistent = doc.get("nonexistent", expected_default)
        assert result_header == expected_header
        assert result_nonexistent == expected_default

    def test_get_recursive_text(self):
        doc = HeaderDocument(id="doc1", text="Content", metadata={
                             "parent_header": "Chapter 1"})
        expected = "Chapter 1\nContent\n"
        result = doc.get_recursive_text()
        assert result == expected


class TestHeaderTextNode:
    def test_init(self):
        expected_id = "node1"
        expected_text = "Node content"
        expected_metadata: HeaderMetadata = {
            "id": expected_id,
            "header": "Section 1",
            "header_level": 2,
            "parent_header": "Chapter 1",
            "doc_index": 0,
            "content": "",
            "chunk_index": None,
            "tokens": None,
            "source_url": None,
            "texts": None,
        }
        node = HeaderTextNode(
            id=expected_id, text=expected_text, metadata=expected_metadata)
        result_id = node.id
        result_text = node.text
        result_metadata = node.metadata
        assert result_id == expected_id
        assert result_text == expected_text
        assert result_metadata == expected_metadata
        assert node.node_id == expected_id

    def test_getitem(self):
        node = HeaderTextNode(id="node1", text="Content",
                              metadata={"header": "Section 1"})
        expected_header = "Section 1"
        result_header = node["header"]
        assert result_header == expected_header

    def test_iter(self):
        node = HeaderTextNode(id="node1", text="Content",
                              metadata={"header": "Section 1"})
        expected_items = {
            "text": "Content",
            "metadata": node.metadata,
            "metadata_template": node.metadata_template,
            "metadata_separator": node.metadata_separator,
            "text_template": node.text_template,
            "id": "node1",
            "header": "Section 1",
            "doc_index": 0,
            "header_level": 0,
            "parent_header": "",
            "content": "",
        }
        result = dict(node)
        for key, expected_value in expected_items.items():
            assert key in result
            assert result[key] == expected_value

    def test_get_content(self):
        node = HeaderTextNode(
            id="node1",
            text="Section 1\nContent",
            metadata={"header": "Section 1", "parent_header": "Chapter 1"}
        )
        expected_all = "Chapter 1\nSection 1\n\nid: node1\ndoc_index: 0\n\nContent"
        expected_none = "Section 1\nContent"
        result_all = node.get_content(metadata_mode=MetadataMode.ALL)
        result_none = node.get_content(metadata_mode=MetadataMode.NONE)
        assert result_all == expected_all
        assert result_none == expected_none

    def test_get_metadata_str(self):
        node = HeaderTextNode(id="node1", text="Content", metadata={
                              "header": "Section 1", "doc_index": 1})
        expected_all = "id: node1\ndoc_index: 1"
        expected_embed = "header: Section 1"
        result_all = node.get_metadata_str(mode=MetadataMode.ALL)
        result_embed = node.get_metadata_str(mode=MetadataMode.EMBED)
        assert result_all == expected_all
        assert result_embed == expected_embed


class TestHeaderDocumentWithScore:
    def setup_header_doc(self) -> HeaderDocument:
        return HeaderDocument(
            id="doc1",
            text="Sample content",
            metadata={
                "header": "Introduction",
                "parent_header": "Chapter 1",
                "header_level": 1,
                "doc_index": 0,
                "content": "Sample content",
                "texts": ["Sample content"],
            }
        )

    def test_init(self):
        header_doc = self.setup_header_doc()
        expected_score = 0.95
        expected_doc_index = 0
        expected_rank = 1
        expected_combined_score = 0.90
        expected_embedding_score = 0.92
        expected_headers = ["Introduction"]
        expected_highlighted_text = "Sample <mark>content</mark>"
        expected_matches: List[Match] = [
            {"word": "content", "start_idx": 7,
                "end_idx": 14, "line": "Sample content"}
        ]
        doc_with_score = HeaderDocumentWithScore(
            node=header_doc,
            score=expected_score,
            doc_index=expected_doc_index,
            rank=expected_rank,
            combined_score=expected_combined_score,
            embedding_score=expected_embedding_score,
            headers=expected_headers,
            highlighted_text=expected_highlighted_text,
            matches=expected_matches
        )
        result_score = doc_with_score.score
        result_doc_index = doc_with_score.doc_index
        result_rank = doc_with_score.rank
        result_combined_score = doc_with_score.combined_score
        result_embedding_score = doc_with_score.embedding_score
        result_headers = doc_with_score.headers
        result_highlighted_text = doc_with_score.highlighted_text
        result_matches = doc_with_score.matches
        assert result_score == expected_score
        assert result_doc_index == expected_doc_index
        assert result_rank == expected_rank
        assert result_combined_score == expected_combined_score
        assert result_embedding_score == expected_embedding_score
        assert result_headers == expected_headers
        assert result_highlighted_text == expected_highlighted_text
        assert result_matches == expected_matches
        assert doc_with_score.node == header_doc

    def test_get_score(self):
        header_doc = self.setup_header_doc()
        doc_with_score = HeaderDocumentWithScore(node=header_doc, score=0.95)
        expected_score = 0.95
        result_score = doc_with_score.get_score()
        assert result_score == expected_score

    def test_get_score_none(self):
        header_doc = self.setup_header_doc()
        doc_with_score = HeaderDocumentWithScore(node=header_doc, score=None)
        expected_score = 0.0
        result_score = doc_with_score.get_score()
        assert result_score == expected_score

    def test_get_score_raise_error(self):
        header_doc = self.setup_header_doc()
        doc_with_score = HeaderDocumentWithScore(node=header_doc, score=None)
        with pytest.raises(ValueError, match="Score not set."):
            doc_with_score.get_score(raise_error=True)

    def test_class_name(self):
        expected = "HeaderDocumentWithScore"
        result = HeaderDocumentWithScore.class_name()
        assert result == expected

    def test_node_id(self):
        header_doc = self.setup_header_doc()
        doc_with_score = HeaderDocumentWithScore(node=header_doc)
        expected_node_id = "doc1"
        result_node_id = doc_with_score.node_id
        assert result_node_id == expected_node_id

    def test_id_(self):
        header_doc = self.setup_header_doc()
        doc_with_score = HeaderDocumentWithScore(node=header_doc)
        expected_id = "doc1"
        result_id = doc_with_score.id_
        assert result_id == expected_id

    def test_text(self):
        header_doc = self.setup_header_doc()
        doc_with_score = HeaderDocumentWithScore(node=header_doc)
        expected_text = "Sample content"
        result_text = doc_with_score.text
        assert result_text == expected_text

    def test_metadata(self):
        header_doc = self.setup_header_doc()
        doc_with_score = HeaderDocumentWithScore(node=header_doc)
        expected_metadata: Dict[str, Any] = {
            "id": "doc1",
            "header": "Introduction",
            "parent_header": "Chapter 1",
            "header_level": 1,
            "doc_index": 0,
            "content": "Sample content",
            "chunk_index": None,
            "tokens": None,
            "source_url": None,
            "links": None,
            "texts": ["Sample content"],
        }
        result_metadata = doc_with_score.metadata
        assert result_metadata == expected_metadata

    def test_getitem_metadata_access(self):
        header_doc = self.setup_header_doc()
        doc_with_score = HeaderDocumentWithScore(node=header_doc)
        expected_header = "Introduction"
        expected_parent_header = "Chapter 1"
        result_header = doc_with_score["header"]
        result_parent_header = doc_with_score["parent_header"]
        assert result_header == expected_header
        assert result_parent_header == expected_parent_header

    def test_iter(self):
        header_doc = self.setup_header_doc()
        doc_with_score = HeaderDocumentWithScore(
            node=header_doc,
            score=0.95,
            doc_index=0,
            rank=1,
            combined_score=0.90,
            embedding_score=0.92,
            headers=["Introduction"],
            highlighted_text="Sample <mark>content</mark>",
            matches=[{"word": "content", "start_idx": 7,
                      "end_idx": 14, "line": "Sample content"}]
        )
        expected_items = {
            "node": header_doc,
            "score": 0.95,
            "doc_index": 0,
            "rank": 1,
            "combined_score": 0.90,
            "embedding_score": 0.92,
            "headers": ["Introduction"],
            "highlighted_text": "Sample <mark>content</mark>",
            "matches": [{"word": "content", "start_idx": 7, "end_idx": 14, "line": "Sample content"}],
            "text": "Sample content",
            "metadata": header_doc.metadata,
            "metadata_separator": header_doc.metadata_separator,
            "id": "doc1",
            "header": "Introduction",
            "parent_header": "Chapter 1",
            "header_level": 1,
            "content": "Sample content",
            "texts": ["Sample content"],
        }
        result = dict(doc_with_score)
        for key, expected_value in expected_items.items():
            assert key in result
            assert result[key] == expected_value

    def test_get(self):
        header_doc = self.setup_header_doc()
        doc_with_score = HeaderDocumentWithScore(node=header_doc, score=0.95)
        expected_header = "Introduction"
        expected_score = 0.95
        expected_default = "default"
        result_header = doc_with_score.get("header")
        result_score = doc_with_score.get("score")
        result_nonexistent = doc_with_score.get(
            "nonexistent", expected_default)
        assert result_header == expected_header
        assert result_score == expected_score
        assert result_nonexistent == expected_default

    def test_get_content(self):
        header_doc = self.setup_header_doc()
        doc_with_score = HeaderDocumentWithScore(node=header_doc)
        expected_none = "Sample content"
        result_none = doc_with_score.get_content(
            metadata_mode=MetadataMode.NONE)
        assert result_none == expected_none

    def test_str(self):
        header_doc = self.setup_header_doc()
        doc_with_score = HeaderDocumentWithScore(
            node=header_doc,
            score=0.95,
            rank=1,
            combined_score=0.90,
            embedding_score=0.92
        )
        expected = (
            f"{str(header_doc)}\n"
            "Score: 0.950\n"
            "Rank: 1\n"
            "Combined Score: 0.900\n"
            "Embedding Score: 0.920"
        )
        result = str(doc_with_score)
        assert result == expected
