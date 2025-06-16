import pytest
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from jet.llm.mlx.tasks.eval.evaluate_multiple_contexts_relevance import (
    evaluate_multiple_contexts_relevance,
    embed_query_context_pairs,
    load_classifier,
    save_classifier,
    ExtendedModelComponents,
    load_model_components,
    InvalidInputError
)
import os
import joblib
from tempfile import TemporaryDirectory


class TestEvaluateMultipleContextsRelevance:
    @pytest.fixture(autouse=True)
    def setup_components(self):
        self.model_path = "mlx-community/Qwen3-1.7B-4bit-DWQ-053125"
        self.classifier, self.label_encoder, self.embedder = load_classifier()
        model_components = load_model_components(self.model_path)
        self.model_components = ExtendedModelComponents(
            model_components.model,
            model_components.tokenizer,
            self.classifier,
            self.label_encoder,
            self.embedder
        )

    def test_highly_relevant_context(self):
        query = "What is the capital of France?"
        contexts = ["The capital of France is Paris."]
        result = evaluate_multiple_contexts_relevance(
            query, contexts, self.model_components
        )['results'][0]
        expected = {
            "relevance_score": 2,
            "confidence": float,
            "is_valid": True,
            "error": None,
            "context": contexts[0]
        }
        assert result["relevance_score"] == expected[
            "relevance_score"], f"Expected score {expected['relevance_score']}, got {result['relevance_score']}"
        assert isinstance(result["confidence"],
                          float), "Confidence should be a float"
        assert result["is_valid"] == expected[
            "is_valid"], f"Expected is_valid {expected['is_valid']}, got {result['is_valid']}"
        assert result["error"] == expected[
            "error"], f"Expected error {expected['error']}, got {result['error']}"
        assert result["context"] == expected[
            "context"], f"Expected context {expected['context']}, got {result['context']}"

    def test_somewhat_relevant_context(self):
        query = "What is the capital of France?"
        contexts = ["Paris is a popular tourist destination."]
        result = evaluate_multiple_contexts_relevance(
            query, contexts, self.model_components
        )['results'][0]
        expected = {
            "relevance_score": 1,
            "confidence": float,
            "is_valid": True,
            "error": None,
            "context": contexts[0]
        }
        assert result["relevance_score"] == expected[
            "relevance_score"], f"Expected score {expected['relevance_score']}, got {result['relevance_score']}"
        assert isinstance(result["confidence"],
                          float), "Confidence should be a float"
        assert result["is_valid"] == expected[
            "is_valid"], f"Expected is_valid {expected['is_valid']}, got {result['is_valid']}"
        assert result["error"] == expected[
            "error"], f"Expected error {expected['error']}, got {result['error']}"
        assert result["context"] == expected[
            "context"], f"Expected context {expected['context']}, got {result['context']}"

    def test_not_relevant_context(self):
        query = "What is the capital of France?"
        contexts = ["Einstein developed the theory of relativity."]
        result = evaluate_multiple_contexts_relevance(
            query, contexts, self.model_components
        )['results'][0]
        expected = {
            "relevance_score": 0,
            "confidence": float,
            "is_valid": True,
            "error": None,
            "context": contexts[0]
        }
        assert result["relevance_score"] == expected[
            "relevance_score"], f"Expected score {expected['relevance_score']}, got {result['relevance_score']}"
        assert isinstance(result["confidence"],
                          float), "Confidence should be a float"
        assert result["is_valid"] == expected[
            "is_valid"], f"Expected is_valid {expected['is_valid']}, got {result['is_valid']}"
        assert result["error"] == expected[
            "error"], f"Expected error {expected['error']}, got {result['error']}"
        assert result["context"] == expected[
            "context"], f"Expected context {expected['context']}, got {result['context']}"

    def test_invalid_input_empty_query(self):
        query = ""
        contexts = ["Valid context"]
        with pytest.raises(InvalidInputError):
            evaluate_multiple_contexts_relevance(
                query, contexts, self.model_components
            )

    def test_invalid_input_empty_context(self):
        query = "Valid query"
        contexts = [""]
        with pytest.raises(InvalidInputError):
            evaluate_multiple_contexts_relevance(
                query, contexts, self.model_components
            )

    def test_invalid_input_empty_contexts(self):
        query = "Valid query"
        contexts = []
        with pytest.raises(InvalidInputError):
            evaluate_multiple_contexts_relevance(
                query, contexts, self.model_components
            )


class TestEmbedQueryContextPairs:
    def test_embedding_shape(self):
        embedder = SentenceTransformer(
            "static-retrieval-mrl-en-v1", device="cpu", backend="onnx")
        pairs = [
            "Query: Test query\nContext: Test context",
            "Query: Another query\nContext: Another context"
        ]
        result = embed_query_context_pairs(pairs, embedder)
        expected_shape = (2, embedder.get_sentence_embedding_dimension())
        assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"

    def test_embedding_values(self):
        embedder = SentenceTransformer(
            "static-retrieval-mrl-en-v1", device="cpu", backend="onnx")
        pairs = ["Query: Test query\nContext: Test context"]
        result = embed_query_context_pairs(pairs, embedder)
        expected_type = np.float32
        expected_shape = (1, embedder.get_sentence_embedding_dimension())
        assert result.dtype == expected_type, f"Expected dtype {expected_type}, got {result.dtype}"
        assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"
        assert not np.any(np.isnan(result)), "Embeddings contain NaN values"
        assert not np.any(
            np.isinf(result)), "Embeddings contain infinite values"


class TestLoadAndSaveClassifier:
    @pytest.fixture
    def classifier_components(self):
        classifier, label_encoder, embedder = load_classifier()
        return classifier, label_encoder, embedder

    def test_load_classifier_default(self):
        result_classifier, result_label_encoder, result_embedder = load_classifier()
        expected_classifier_type = LogisticRegression
        expected_label_encoder_type = LabelEncoder
        expected_embedder_type = SentenceTransformer
        assert isinstance(
            result_classifier, expected_classifier_type), f"Expected classifier type {expected_classifier_type}, got {type(result_classifier)}"
        assert isinstance(
            result_label_encoder, expected_label_encoder_type), f"Expected label encoder type {expected_label_encoder_type}, got {type(result_label_encoder)}"
        assert isinstance(
            result_embedder, expected_embedder_type), f"Expected embedder type {expected_embedder_type}, got {type(result_embedder)}"

    def test_save_and_load_classifier(self, classifier_components):
        classifier, label_encoder, embedder = classifier_components
        with TemporaryDirectory() as temp_dir:
            # Save classifier components
            save_classifier(classifier, label_encoder, embedder,
                            save_dir=temp_dir, verbose=False)

            # Verify files exist
            expected_files = ["classifier.joblib",
                              "label_encoder.joblib", "embedder.joblib"]
            for file_name in expected_files:
                assert os.path.exists(os.path.join(
                    temp_dir, file_name)), f"File {file_name} not saved"

            # Load classifier components
            result_classifier, result_label_encoder, result_embedder = load_classifier(
                save_dir=temp_dir, verbose=False)

            # Verify types
            assert isinstance(
                result_classifier, LogisticRegression), f"Loaded classifier is not {LogisticRegression}"
            assert isinstance(
                result_label_encoder, LabelEncoder), f"Loaded label encoder is not {LabelEncoder}"
            assert isinstance(
                result_embedder, SentenceTransformer), f"Loaded embedder is not {SentenceTransformer}"

            # Verify classifier can predict (functional test)
            test_pairs = ["Query: Test query\nContext: Test context"]
            embeddings = embed_query_context_pairs(test_pairs, result_embedder)
            result_pred = result_classifier.predict(embeddings)
            expected_pred_shape = (1,)
            assert result_pred.shape == expected_pred_shape, f"Expected prediction shape {expected_pred_shape}, got {result_pred.shape}"

    def test_save_classifier_none_dir(self, classifier_components):
        classifier, label_encoder, embedder = classifier_components
        # Should not raise an error and skip saving
        save_classifier(classifier, label_encoder, embedder,
                        save_dir=None, verbose=False)
        # No files should be created, so no further assertions needed
