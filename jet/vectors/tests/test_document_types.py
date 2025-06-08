import pytest
import uuid
from typing import Any, Dict
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from jet.vectors.document_types import HeaderMetadata, HeaderDocument, HeaderTextNode


class TestHeaderDocumentIdConsistency:
    def test_id_equals_id_no_input(self):
        """Test that id, id_, and metadata['id'] are consistent with UUID when no id is provided."""
        doc = HeaderDocument(text="Test")
        expected_id = doc.id_  # Pydantic-generated UUID
        result_id = doc.id
        result_metadata_id = doc.metadata["id"]

        assert result_id == expected_id, f"Expected self.id {result_id} to equal self.id_ {expected_id}"
        assert result_metadata_id == expected_id, f"Expected metadata['id'] {result_metadata_id} to equal self.id_ {expected_id}"
        assert len(
            expected_id) == 36, f"Expected UUID length 36, got {len(expected_id)}"

    def test_id_equals_id_with_custom_id(self):
        """Test that id, id_, and metadata['id'] are consistent with custom id."""
        expected_id = "custom-id"
        doc = HeaderDocument(id=expected_id, text="Test", metadata={
                             "id": "wrong-id", "header": "H1"})
        result_id = doc.id
        result_id_ = doc.id_
        result_metadata_id = doc.metadata["id"]

        assert result_id == expected_id, f"Expected self.id {result_id} to equal {expected_id}"
        assert result_id_ == expected_id, f"Expected self.id_ {result_id_} to equal {expected_id}"
        assert result_metadata_id == expected_id, f"Expected metadata['id'] {result_metadata_id} to equal {expected_id}"


class TestHeaderTextNodeIdConsistency:
    def test_id_equals_id_no_input(self):
        """Test that id, id_, and metadata['id'] are consistent with UUID when no id is provided."""
        node = HeaderTextNode(text="Test")
        expected_id = node.id_  # Pydantic-generated UUID
        result_id = node.id
        result_metadata_id = node.metadata["id"]

        assert result_id == expected_id, f"Expected self.id {result_id} to equal self.id_ {expected_id}"
        assert result_metadata_id == expected_id, f"Expected metadata['id'] {result_metadata_id} to equal self.id_ {expected_id}"
        assert len(
            expected_id) == 36, f"Expected UUID length 36, got {len(expected_id)}"

    def test_id_equals_id_with_custom_id(self):
        """Test that id, id_, and metadata['id'] are consistent with custom id."""
        expected_id = "custom-id"
        node = HeaderTextNode(id=expected_id, text="Test", metadata={
                              "id": "wrong-id", "header": "H1"})
        result_id = node.id
        result_id_ = node.id_
        result_metadata_id = node.metadata["id"]

        assert result_id == expected_id, f"Expected self.id {result_id} to equal {expected_id}"
        assert result_id_ == expected_id, f"Expected self.id_ {result_id_} to equal {expected_id}"
        assert result_metadata_id == expected_id, f"Expected metadata['id'] {result_metadata_id} to equal {expected_id}"
