import pytest
import torch
from typing import List, Optional
from unittest.mock import Mock, patch
from pydantic import BaseModel
import uuid
from jet.models.tasks.llm_rerank import RerankResult, format_instruction, process_inputs, compute_logits, rerank_docs


class TestRerankDocs:
    @pytest.fixture
    def mock_tokenizer(self):
        tokenizer = Mock()
        tokenizer.encode.side_effect = lambda x, add_special_tokens=False: [
            1, 2, 3] if x != "no" and x != "yes" else [4] if x == "no" else [5]
        tokenizer.convert_tokens_to_ids.side_effect = lambda x: 4 if x == "no" else 5
        tokenizer.pad.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]])
        }
        return tokenizer

    @pytest.fixture
    def mock_model(self):
        model = Mock()
        model.return_value.logits = torch.tensor(
            [[[0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 2.0]]])
        return model

    def test_rerank_docs_with_ids(self, mock_tokenizer, mock_model):
        queries = ["What is the capital?", "What is gravity?"]
        documents = ["Capital is Paris.", "Gravity is a force."]
        ids = ["doc1", "doc2"]
        instruction = "Test instruction"

        expected = [
            RerankResult(
                id="doc1",
                rank=2,
                doc_index=0,
                score=pytest.approx(0.11920292, rel=1e-5),
                text="Capital is Paris.",
                tokens=3
            ),
            RerankResult(
                id="doc2",
                rank=1,
                doc_index=1,
                score=pytest.approx(0.88079708, rel=1e-5),
                text="Gravity is a force.",
                tokens=3
            )
        ]

        with patch("rerank_docs.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
            with patch("rerank_docs.AutoModelForCausalLM.from_pretrained", return_value=mock_model):
                result = rerank_docs(queries, documents,
                                     ids=ids, instruction=instruction)

        for r, e in zip(result, expected):
            assert r.id == e.id
            assert r.rank == e.rank
            assert r.doc_index == e.doc_index
            assert r.score == pytest.approx(e.score, rel=1e-5)
            assert r.text == e.text
            assert r.tokens == e.tokens

    def test_rerank_docs_without_ids(self, mock_tokenizer, mock_model):
        queries = ["What is the capital?", "What is gravity?"]
        documents = ["Capital is Paris.", "Gravity is a force."]
        instruction = "Test instruction"

        with patch("rerank_docs.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
            with patch("rerank_docs.AutoModelForCausalLM.from_pretrained", return_value=mock_model):
                result = rerank_docs(queries, documents,
                                     instruction=instruction)

        expected_ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        expected = [
            RerankResult(
                id=expected_ids[0],
                rank=2,
                doc_index=0,
                score=pytest.approx(0.11920292, rel=1e-5),
                text="Capital is Paris.",
                tokens=3
            ),
            RerankResult(
                id=expected_ids[1],
                rank=1,
                doc_index=1,
                score=pytest.approx(0.88079708, rel=1e-5),
                text="Gravity is a force.",
                tokens=3
            )
        ]

        for r, e in zip(result, expected):
            assert uuid.UUID(r.id)  # Verify valid UUID
            assert r.rank == e.rank
            assert r.doc_index == e.doc_index
            assert r.score == pytest.approx(e.score, rel=1e-5)
            assert r.text == e.text
            assert r.tokens == e.tokens

    def test_rerank_docs_invalid_ids_length(self, mock_tokenizer, mock_model):
        queries = ["What is the capital?", "What is gravity?"]
        documents = ["Capital is Paris.", "Gravity is a force."]
        ids = ["doc1"]  # Length mismatch

        expected_error = "Length of ids must match length of documents"

        with patch("rerank_docs.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
            with patch("rerank_docs.AutoModelForCausalLM.from_pretrained", return_value=mock_model):
                with pytest.raises(ValueError, match=expected_error):
                    rerank_docs(queries, documents, ids=ids)

    def test_format_instruction_with_instruction(self):
        instruction = "Test instruction"
        query = "What is the capital?"
        doc = "Capital is Paris."

        expected = "<Instruct>: Test instruction\n<Query>: What is the capital?\n<Document>: Capital is Paris."
        result = format_instruction(instruction, query, doc)

        assert result == expected

    def test_format_instruction_without_instruction(self):
        query = "What is the capital?"
        doc = "Capital is Paris."

        expected = "<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n<Query>: What is the capital?\n<Document>: Capital is Paris."
        result = format_instruction(None, query, doc)

        assert result == expected
