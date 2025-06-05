import pytest
import torch
import numpy as np
from unittest.mock import patch, Mock
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, AutoConfig
from span_marker import SpanMarkerModel
from jet.logger import logger
from jet.llm.embeddings.fast_embedding import EmbeddingGenerator


class TestEmbeddingGenerator:
    """Test suite for EmbeddingGenerator class."""

    def setup_method(self):
        """Setup common test fixtures."""
        self.model_name = "all-MiniLM-L6-v2"
        self.texts = ["This is a test sentence.", "Another test sentence."]
        self.batch_size = 2
        self.max_length = 128

    @patch("torch.backends.mps.is_available")
    def test_init_sentence_transformer(self, mock_mps):
        """Test initialization with sentence transformer model."""
        mock_mps.return_value = False
        expected_device = "cpu"
        expected_model_type = "sentence_transformer"

        generator = EmbeddingGenerator(self.model_name, use_mps=False)
        result_device = generator.device
        result_model_type = generator.model_type
        result_model = isinstance(generator.model, SentenceTransformer)
        result_tokenizer = generator.tokenizer

        assert result_device == expected_device
        assert result_model_type == expected_model_type
        assert result_model is True
        assert result_tokenizer is None

    @patch("torch.backends.mps.is_available")
    @patch("transformers.AutoConfig.from_pretrained")
    def test_detect_model_type_causal(self, mock_config, mock_mps):
        """Test model type detection for causal model."""
        mock_mps.return_value = False
        mock_config.return_value = Mock(model_type="gpt2")
        model_name = "gpt2"
        expected_model_type = "causal"

        generator = EmbeddingGenerator(model_name)
        result_model_type = generator.model_type

        assert result_model_type == expected_model_type

    @patch("torch.backends.mps.is_available")
    @patch("transformers.AutoConfig.from_pretrained")
    def test_detect_model_type_fallback(self, mock_config, mock_mps):
        """Test model type detection fallback to sentence_transformer."""
        mock_mps.return_value = False
        mock_config.side_effect = Exception("Config error")
        expected_model_type = "sentence_transformer"

        with patch.object(logger, "warning") as mock_logger:
            generator = EmbeddingGenerator(self.model_name)
            result_model_type = generator.model_type

            assert result_model_type == expected_model_type
            mock_logger.assert_called_once()

    @patch("torch.backends.mps.is_available")
    @patch("sentence_transformers.SentenceTransformer")
    def test_load_model_sentence_transformer(self, mock_st, mock_mps):
        """Test loading sentence transformer model."""
        mock_mps.return_value = False
        mock_model = Mock(spec=SentenceTransformer)
        mock_st.return_value = mock_model
        expected_model = mock_model
        expected_tokenizer = None

        generator = EmbeddingGenerator(
            self.model_name, model_type="sentence_transformer")
        result_model = generator.model
        result_tokenizer = generator.tokenizer

        assert result_model == expected_model
        assert result_tokenizer == expected_tokenizer

    def test_batch_iterator(self):
        """Test batch iterator splits texts correctly."""
        generator = EmbeddingGenerator(self.model_name)
        texts = ["text1", "text2", "text3", "text4"]
        batch_size = 2
        expected_batches = [["text1", "text2"], ["text3", "text4"]]

        result_batches = list(generator._batch_iterator(texts, batch_size))

        assert result_batches == expected_batches

    @patch("torch.backends.mps.is_available")
    @patch("transformers.AutoModel.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_encode_causal(self, mock_tokenizer, mock_model, mock_mps):
        """Test causal model encoding."""
        mock_mps.return_value = False
        mock_model.return_value = Mock()
        mock_tokenizer.return_value = Mock(pad_token=None, eos_token="[EOS]")
        mock_model.return_value.return_value.last_hidden_state = torch.ones(
            (2, 10, 768))
        generator = EmbeddingGenerator("gpt2", model_type="causal")
        texts = self.texts
        expected_shape = (2, 768)
        expected_embeddings = np.ones(expected_shape)

        result_embeddings = generator._encode_causal(
            texts, self.max_length, self.batch_size, normalize=False)

        assert result_embeddings.shape == expected_shape
        assert np.allclose(result_embeddings, expected_embeddings, atol=1e-5)

    @patch("torch.backends.mps.is_available")
    @patch("transformers.AutoModel.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_encode_transformer(self, mock_tokenizer, mock_model, mock_mps):
        """Test transformer model encoding with mean pooling."""
        mock_mps.return_value = False
        mock_model.return_value = Mock()
        mock_tokenizer.return_value = Mock()
        mock_model.return_value.return_value.last_hidden_state = torch.ones(
            (2, 10, 768))
        mock_tokenizer.return_value.return_value = {
            "input_ids": torch.ones((2, 10)),
            "attention_mask": torch.ones((2, 10))
        }
        generator = EmbeddingGenerator("bert-base-uncased", model_type="auto")
        texts = self.texts
        expected_shape = (2, 768)
        expected_embeddings = np.ones(expected_shape)

        result_embeddings = generator._encode_transformer(
            texts, self.max_length, self.batch_size, normalize=False)

        assert result_embeddings.shape == expected_shape
        assert np.allclose(result_embeddings, expected_embeddings, atol=1e-5)

    @patch("torch.backends.mps.is_available")
    @patch("span_marker.SpanMarkerModel.from_pretrained")
    def test_encode_span_marker(self, mock_span_marker, mock_mps):
        """Test span marker model encoding."""
        mock_mps.return_value = False
        mock_model = Mock(spec=SpanMarkerModel)
        mock_model.predict.return_value = [
            [{"span": "entity", "label": "PER", "score": 0.9, "char_start_index": 0, "char_end_index": 6}]]
        mock_span_marker.return_value = mock_model
        generator = EmbeddingGenerator(
            "span-marker-model", model_type="span_marker")
        texts = ["Test entity"]
        expected_results = [[{"span": "entity", "label": "PER",
                              "score": 0.9, "char_start_index": 0, "char_end_index": 6}]]

        result_entities = generator._encode_span_marker(
            texts, self.max_length, self.batch_size, normalize=False)

        assert result_entities == expected_results

    @patch("torch.backends.mps.is_available")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_chunk_text(self, mock_tokenizer, mock_mps):
        """Test text chunking with tokenizer."""
        mock_mps.return_value = False
        mock_tokenizer.return_value.encode_plus.return_value = {
            "input_ids": [101, 102, 103, 104, 105],
            "offset_mapping": [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20)]
        }
        mock_tokenizer.return_value.decode.return_value = "chunked text"

        generator = EmbeddingGenerator("bert-base-uncased", model_type="auto")
        text = "This is a long text to chunk"

        expected_chunks = [
            {
                "text": "chunked text",
                "input_ids": [101, 102, 103, 104, 105],
                "offset_mapping": [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20)],
                "char_start": 0,
                "char_end": 20
            },
            {
                "text": "chunked text",
                "input_ids": [103, 104, 105],
                "offset_mapping": [(8, 12), (12, 16), (16, 20)],
                "char_start": 8,
                "char_end": 20
            },
            {
                "text": "chunked text",
                "input_ids": [105],
                "offset_mapping": [(16, 20)],
                "char_start": 16,
                "char_end": 20
            }
        ]

        result_chunks = generator.chunk_text(text, max_length=5, stride=3)

        assert result_chunks == expected_chunks

    @patch("torch.backends.mps.is_available")
    @patch("sentence_transformers.SentenceTransformer")
    def test_generate_embeddings_sentence_transformer(self, mock_st, mock_mps):
        """Test embedding generation with sentence transformer."""
        mock_mps.return_value = False
        mock_model = Mock(spec=SentenceTransformer)
        mock_model.encode.return_value = np.ones((2, 384))
        mock_st.return_value = mock_model
        generator = EmbeddingGenerator(
            self.model_name, model_type="sentence_transformer")
        texts = self.texts
        expected_shape = (2, 384)
        expected_embeddings = np.ones(expected_shape)

        result_embeddings = generator.generate_embeddings(
            texts, self.batch_size, self.max_length, normalize=False)

        assert result_embeddings.shape == expected_shape
        assert np.allclose(result_embeddings, expected_embeddings, atol=1e-5)

    def test_generate_embeddings_empty_input(self):
        """Test embedding generation with empty document list."""
        generator = EmbeddingGenerator(self.model_name)

        with pytest.raises(ValueError) as exc_info:
            generator.generate_embeddings([], self.batch_size, self.max_length)

        expected_error = "Document list cannot be empty"
        result_error = str(exc_info.value)

        assert result_error == expected_error
