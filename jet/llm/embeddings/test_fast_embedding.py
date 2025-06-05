import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from jet.llm.embeddings.fast_embedding import EmbeddingGenerator
from jet.llm.embeddings.sentence_embedding import SentenceEmbedding


class TestEmbeddingGenerator:
    """Test suite for EmbeddingGenerator class."""

    @pytest.fixture
    def sentence_transformer_generator(self):
        """Fixture for SentenceEmbedding model."""
        with patch('jet.llm.embeddings.sentence_embedding.SentenceEmbedding') as mock_model:
            generator = EmbeddingGenerator(
                model_name='all-MiniLM-L6-v2', model_type='sentence_transformer', use_mps=False)
            mock_model.return_value.encode.return_value = np.array(
                [[0.1, 0.2], [0.3, 0.4]])
            return generator

    @pytest.fixture
    def span_marker_generator(self):
        """Fixture for SpanMarker model."""
        with patch('jet.llm.embeddings.fast_embedding.SpanMarkerModel') as mock_model:
            with patch('jet.llm.embeddings.fast_embedding.SpanMarkerTokenizer') as mock_tokenizer:
                with patch('jet.llm.embeddings.fast_embedding.AutoConfig') as mock_config:
                    mock_config.return_value.model_max_length = 512
                    generator = EmbeddingGenerator(
                        model_name='tomaarsen/span-marker-bert-base-fewnerd-fine-super',
                        model_type='span_marker',
                        use_mps=False
                    )
                    mock_model.return_value.predict.return_value = [
                        [{"span": "John", "label": "PERSON", "score": 0.9,
                            "char_start_index": 0, "char_end_index": 4}],
                        [{"span": "Paris", "label": "LOCATION", "score": 0.95,
                            "char_start_index": 0, "char_end_index": 5}]
                    ]
                    return generator

    def test_init_sentence_transformer(self, sentence_transformer_generator):
        """Test initialization with sentence transformer model."""
        expected_device = 'cpu'
        expected_model_type = 'sentence_transformer'
        result_device = sentence_transformer_generator.device
        result_model_type = sentence_transformer_generator.model_type

        assert result_device == expected_device, f"Expected device {expected_device}, got {result_device}"
        assert result_model_type == expected_model_type, f"Expected model_type {expected_model_type}, got {result_model_type}"

    def test_generate_embeddings_sentence_transformer(self, sentence_transformer_generator):
        """Test embedding generation for sentence transformer."""
        documents = ["test sentence 1", "test sentence 2"]
        expected_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
        result_embeddings = sentence_transformer_generator.generate_embeddings(
            documents, batch_size=2, max_length=128, normalize=False)

        np.testing.assert_array_almost_equal(result_embeddings, expected_embeddings, decimal=5,
                                             err_msg="Embeddings do not match expected values")

    def test_generate_embeddings_empty_input(self, sentence_transformer_generator):
        """Test embedding generation with empty document list."""
        documents = []
        expected_error = ValueError
        with pytest.raises(expected_error, match="Document list cannot be empty"):
            sentence_transformer_generator.generate_embeddings(documents)

    @patch('jet.llm.embeddings.fast_embedding.AutoModel')
    @patch('jet.llm.embeddings.fast_embedding.AutoTokenizer')
    def test_init_causal_model(self, mock_tokenizer, mock_model):
        """Test initialization with causal model."""
        mock_tokenizer.return_value.pad_token = None
        mock_tokenizer.return_value.eos_token = '</s>'
        generator = EmbeddingGenerator(
            model_name='gpt2', model_type='causal', use_mps=False)

        expected_model_type = 'causal'
        expected_device = 'cpu'
        result_model_type = generator.model_type
        result_device = generator.device

        assert result_model_type == expected_model_type, f"Expected model_type {expected_model_type}, got {result_model_type}"
        assert result_device == expected_device, f"Expected device {expected_device}, got {result_device}"
        assert mock_tokenizer.return_value.pad_token == '</s>', "Pad token not set correctly"

    def test_init_span_marker(self, span_marker_generator):
        """Test initialization with span marker model."""
        expected_device = 'cpu'
        expected_model_type = 'span_marker'
        result_device = span_marker_generator.device
        result_model_type = span_marker_generator.model_type

        assert result_device == expected_device, f"Expected device {expected_device}, got {result_device}"
        assert result_model_type == expected_model_type, f"Expected model_type {expected_model_type}, got {result_model_type}"
        assert span_marker_generator.tokenizer is not None, "Tokenizer should be initialized for span_marker"

    def test_generate_embeddings_span_marker(self, span_marker_generator):
        """Test entity prediction for span marker model."""
        documents = ["John lives in Paris", "Paris is a city"]
        expected_entities = [
            [{"span": "John", "label": "PERSON", "score": 0.9,
                "char_start_index": 0, "char_end_index": 4}],
            [{"span": "Paris", "label": "LOCATION", "score": 0.95,
                "char_start_index": 0, "char_end_index": 5}]
        ]
        result_entities = span_marker_generator.generate_embeddings(
            documents, batch_size=2, max_length=128, normalize=False)

        assert result_entities == expected_entities, f"Expected entities {expected_entities}, got {result_entities}"
