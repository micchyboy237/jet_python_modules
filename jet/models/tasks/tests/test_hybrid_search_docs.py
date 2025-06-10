import pytest
from typing import List, Any, Optional
from jet.vectors.document_types import HeaderDocument
from jet.models.tasks.hybrid_search_docs_with_bm25 import get_original_document


class TestGetOriginalDocument:
    def test_mapping_with_ids_for_strings(self):
        # Arrange
        documents = ["text1", "text2"]
        ids = ["id1", "id2"]
        doc_id = "id1"
        doc_index = 0
        expected = HeaderDocument(
            id="id1", text="text1", metadata={"original_index": 0})

        # Act
        result = get_original_document(doc_id, doc_index, documents, ids)

        # Assert
        assert result == expected, f"Expected {expected}, got {result}"

    def test_mapping_without_ids_for_strings(self):
        # Arrange
        documents = ["text1", "text2", "text3"]
        doc_id = "doc_1"  # Matches index 1
        doc_index = 1
        expected = HeaderDocument(
            id="doc_1", text="text2", metadata={"original_index": 1})

        # Act
        result = get_original_document(doc_id, doc_index, documents, None)

        # Assert
        assert result == expected, f"Expected {expected}, got {result}"

    def test_mapping_without_ids_using_index(self):
        # Arrange
        documents = ["text1", "text2", "text3"]
        doc_id = "invalid_id"  # Should match by index
        doc_index = 2
        expected = HeaderDocument(
            id="doc_2", text="text3", metadata={"original_index": 2})

        # Act
        result = get_original_document(doc_id, doc_index, documents, None)

        # Assert
        assert result == expected, f"Expected {expected}, got {result}"

    def test_no_document_found(self):
        # Arrange
        documents = ["text1"]
        doc_id = "doc_999"
        doc_index = 999
        expected = None

        # Act
        result = get_original_document(doc_id, doc_index, documents, None)

        # Assert
        assert result == expected, f"Expected {expected}, got {result}"

    def test_empty_documents(self):
        # Arrange
        documents = []
        doc_id = "doc_0"
        doc_index = 0
        expected = None

        # Act
        result = get_original_document(doc_id, doc_index, documents, None)

        # Assert
        assert result == expected, f"Expected {expected}, got {result}"
