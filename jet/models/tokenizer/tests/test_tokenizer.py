import pytest
import logging
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
from typing import List, Union

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerBase

from jet.models.model_types import ModelType
from jet.models.config import MODELS_CACHE_DIR
from jet.models.tokenizer import get_tokenizer, _tokenizer_cache

# Set up logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def clear_cache():
    """Clear the tokenizer cache before and after each test."""
    _tokenizer_cache.clear()
    yield
    _tokenizer_cache.clear()


@pytest.fixture
def mock_tokenizer():
    """Mock a Tokenizer instance with all required methods."""
    tokenizer = Mock(spec=Tokenizer)
    tokenizer.encode.return_value = Mock(ids=[101, 102, 103])
    tokenizer.decode.return_value = "test text"
    tokenizer.convert_ids_to_tokens.return_value = ["test", "text"]
    # Mock __call__ to simulate encode behavior
    tokenizer.__call__ = Mock(return_value=Mock(ids=[101, 102, 103]))
    logger.debug(f"Mock Tokenizer created with methods: {dir(tokenizer)}")
    return tokenizer


@pytest.fixture
def mock_pretrained_tokenizer():
    """Mock a PreTrainedTokenizerBase instance with all required methods."""
    tokenizer = Mock(spec=PreTrainedTokenizerBase)
    tokenizer.encode.return_value = [101, 102, 103]
    tokenizer.decode.return_value = "test text"
    tokenizer.convert_ids_to_tokens.return_value = ["test", "text"]
    # Mock __call__ to simulate encode behavior
    tokenizer.__call__ = Mock(return_value=[101, 102, 103])
    logger.debug(
        f"Mock PreTrainedTokenizerBase created with methods: {dir(tokenizer)}")
    return tokenizer


class TestGetTokenizer:
    """Tests for the get_tokenizer function."""

    def test_tokenizer_callable_encoding(self, clear_cache, mock_tokenizer, mock_pretrained_tokenizer):
        """Test that tokenizer can be called as a function for encoding."""
        # Given
        model_name: ModelType = "mxbai-embed-large"
        expected_model_path = "mixedbread-ai/mxbai-embed-large-v1"
        expected_text = "test text"
        expected_token_ids: List[int] = [101, 102, 103]
        logger.debug(f"Testing callable encoding for model: {model_name}")

        # When (Test Tokenizer callable)
        with patch("jet.models.utils.resolve_model_value", return_value=expected_model_path), \
                patch("jet.models.tokenizer.Tokenizer.from_pretrained", return_value=mock_tokenizer):
            tokenizer = get_tokenizer(model_name)
            logger.debug(
                f"Tokenizer instance: {type(tokenizer)}, methods: {dir(tokenizer)}")
            result = tokenizer(expected_text)
            logger.debug(f"Tokenizer callable result: {result}")
            result_ids = result.ids
            logger.debug(f"Tokenizer callable result IDs: {result_ids}")

        # Then
        assert result_ids == expected_token_ids, f"Tokenizer callable should return expected IDs, got {result_ids}"

        # When (Test PreTrainedTokenizerBase callable)
        with patch("jet.models.utils.resolve_model_value", return_value=expected_model_path), \
                patch("jet.models.tokenizer.PreTrainedTokenizerBase.from_pretrained", return_value=mock_pretrained_tokenizer), \
                patch("jet.models.tokenizer.Tokenizer.from_pretrained", side_effect=Exception("Remote Tokenizer failed")):
            pretrained_tokenizer = get_tokenizer(model_name)
            logger.debug(
                f"PreTrainedTokenizerBase instance: {type(pretrained_tokenizer)}, methods: {dir(pretrained_tokenizer)}")
            result_ids = pretrained_tokenizer(expected_text)
            logger.debug(
                f"PreTrainedTokenizerBase callable result IDs: {result_ids}")

        # Then
        assert result_ids == expected_token_ids, f"PreTrainedTokenizerBase callable should return expected IDs, got {result_ids}"

    def test_load_tokenizer_from_remote_success(self, clear_cache, mock_tokenizer):
        """Test loading a Tokenizer from remote successfully."""
        # Given
        model_name: ModelType = "mxbai-embed-large"
        expected_model_path = "mixedbread-ai/mxbai-embed-large-v1"
        expected_tokenizer = mock_tokenizer

        # When
        with patch("jet.models.utils.resolve_model_value", return_value=expected_model_path), \
                patch("jet.models.tokenizer.Tokenizer.from_pretrained", return_value=expected_tokenizer):
            result = get_tokenizer(model_name)

        # Then
        assert result == expected_tokenizer, "Expected the mocked tokenizer to be returned"
        assert expected_model_path in _tokenizer_cache, "Tokenizer should be cached"
        assert _tokenizer_cache[expected_model_path] == expected_tokenizer, "Cached tokenizer should match"

    def test_load_pretrained_tokenizer_from_remote_success(self, clear_cache, mock_pretrained_tokenizer):
        """Test loading a PreTrainedTokenizerBase from remote successfully."""
        # Given
        model_name: ModelType = "bert-base-uncased"
        expected_model_path = "bert-base-uncased"
        expected_tokenizer = mock_pretrained_tokenizer

        # When
        with patch("jet.models.utils.resolve_model_value", return_value=expected_model_path), \
                patch("jet.models.tokenizer.PreTrainedTokenizerBase.from_pretrained", return_value=expected_tokenizer), \
                patch("jet.models.tokenizer.Tokenizer.from_pretrained", side_effect=Exception("Remote Tokenizer failed")):
            result = get_tokenizer(model_name)

        # Then
        assert result == expected_tokenizer, "Expected the mocked PreTrainedTokenizerBase to be returned"
        assert expected_model_path in _tokenizer_cache, "PreTrainedTokenizerBase should be cached"
        assert _tokenizer_cache[expected_model_path] == expected_tokenizer, "Cached PreTrainedTokenizerBase should match"

    def test_use_cached_tokenizer(self, clear_cache, mock_tokenizer):
        """Test retrieving a tokenizer from cache."""
        # Given
        model_name: ModelType = "mxbai-embed-large"
        expected_model_path = "mixedbread-ai/mxbai-embed-large-v1"
        _tokenizer_cache[expected_model_path] = mock_tokenizer
        expected_tokenizer = mock_tokenizer

        # When
        with patch("jet.models.utils.resolve_model_value", return_value=expected_model_path), \
                patch("jet.models.tokenizer.Tokenizer.from_pretrained") as mock_from_pretrained:
            result = get_tokenizer(model_name)

        # Then
        assert result == expected_tokenizer, "Expected the cached tokenizer to be returned"
        mock_from_pretrained.assert_not_called(
        ), "Remote loading should not be called when cached"

    def test_fallback_to_local_cache(self, clear_cache, mock_tokenizer):
        """Test falling back to local cache when remote loading fails."""
        # Given
        model_name: ModelType = "mxbai-embed-large"
        expected_model_path = "mixedbread-ai/mxbai-embed-large-v1"
        local_cache_dir = "/mock/cache"
        snapshot_dir = Path(
            local_cache_dir) / f"models--{expected_model_path.replace('/', '--')}" / "snapshots"
        tokenizer_path = snapshot_dir / "hash" / "tokenizer.json"
        expected_tokenizer = mock_tokenizer

        # When
        with patch("jet.models.utils.resolve_model_value", return_value=expected_model_path), \
                patch("jet.models.tokenizer.PreTrainedTokenizerBase.from_pretrained", side_effect=Exception("Remote failed")), \
                patch("jet.models.tokenizer.Tokenizer.from_pretrained", side_effect=Exception("Remote failed")), \
                patch("pathlib.Path.exists", return_value=True), \
                patch("pathlib.Path.rglob", return_value=[tokenizer_path]), \
                patch("pathlib.Path.resolve", return_value=tokenizer_path), \
                patch("pathlib.Path.is_file", return_value=True), \
                patch("jet.models.tokenizer.Tokenizer.from_file", return_value=expected_tokenizer):
            result = get_tokenizer(model_name, local_cache_dir=local_cache_dir)

        # Then
        assert result == expected_tokenizer, "Expected the local tokenizer to be returned"
        assert expected_model_path in _tokenizer_cache, "Tokenizer should be cached"
        assert _tokenizer_cache[expected_model_path] == expected_tokenizer, "Cached tokenizer should match"

    def test_invalid_model_name_raises_value_error(self, clear_cache):
        """Test that an invalid model name raises ValueError."""
        # Given
        model_name: ModelType = "invalid-model"
        expected_model_path = "invalid-model"

        # When
        with patch("jet.models.utils.resolve_model_value", return_value=expected_model_path), \
                patch("jet.models.tokenizer.PreTrainedTokenizerBase.from_pretrained", side_effect=Exception("Remote failed")), \
                patch("jet.models.tokenizer.Tokenizer.from_pretrained", side_effect=Exception("Remote failed")), \
                patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(ValueError) as exc_info:
                get_tokenizer(model_name)

        # Then
        expected_error = "ValueError"
        assert exc_info.typename == expected_error

    def test_local_cache_no_tokenizer_file_raises_value_error(self, clear_cache):
        """Test that no tokenizer.json in local cache raises ValueError."""
        # Given
        model_name: ModelType = "mxbai-embed-large"
        expected_model_path = "mixedbread-ai/mxbai-embed-large-v1"
        local_cache_dir = "/mock/cache"
        snapshot_dir = Path(
            local_cache_dir) / f"models--{expected_model_path.replace('/', '--')}" / "snapshots"

        # When
        with patch("jet.models.utils.resolve_model_value", return_value=expected_model_path), \
                patch("jet.models.tokenizer.PreTrainedTokenizerBase.from_pretrained", side_effect=Exception("Remote failed")), \
                patch("jet.models.tokenizer.Tokenizer.from_pretrained", side_effect=Exception("Remote failed")), \
                patch("pathlib.Path.exists", return_value=True), \
                patch("pathlib.Path.rglob", return_value=[]):
            with pytest.raises(ValueError) as exc_info:
                get_tokenizer(model_name, local_cache_dir=local_cache_dir)

        # Then
        expected_error = f"Could not load tokenizer for {expected_model_path} from remote or local cache."
        assert str(
            exc_info.value) == expected_error, "Expected ValueError with specific message"

    def test_tokenizer_features(self, clear_cache, mock_tokenizer, mock_pretrained_tokenizer):
        """Test that the tokenizer supports encode, decode, and convert_ids_to_tokens."""
        # Given
        model_name: ModelType = "mxbai-embed-large"
        expected_model_path = "mixedbread-ai/mxbai-embed-large-v1"
        expected_text = "test text"
        expected_token_ids = [101, 102, 103]
        expected_tokens = ["test", "text"]
        expected_decoded = "test text"

        # When (Test Tokenizer)
        with patch("jet.models.utils.resolve_model_value", return_value=expected_model_path), \
                patch("jet.models.tokenizer.Tokenizer.from_pretrained", return_value=mock_tokenizer):
            tokenizer = get_tokenizer(model_name)
            result_ids = tokenizer.encode(expected_text).ids
            result_tokens = tokenizer.convert_ids_to_tokens(expected_token_ids)
            result_decoded = tokenizer.decode(expected_token_ids)

        # Then
        assert result_ids == expected_token_ids, "Tokenizer encode should return expected IDs"
        assert result_tokens == expected_tokens, "Tokenizer convert_ids_to_tokens should return expected tokens"
        assert result_decoded == expected_decoded, "Tokenizer decode should return expected text"

        # When (Test PreTrainedTokenizerBase)
        with patch("jet.models.utils.resolve_model_value", return_value=expected_model_path), \
                patch("jet.models.tokenizer.PreTrainedTokenizerBase.from_pretrained", return_value=mock_pretrained_tokenizer), \
                patch("jet.models.tokenizer.Tokenizer.from_pretrained", side_effect=Exception("Remote Tokenizer failed")):
            pretrained_tokenizer = get_tokenizer(model_name)
            result_ids = pretrained_tokenizer.encode(expected_text)
            result_tokens = pretrained_tokenizer.convert_ids_to_tokens(
                expected_token_ids)
            result_decoded = pretrained_tokenizer.decode(expected_token_ids)

        # Then
        assert result_ids == expected_token_ids, "PreTrainedTokenizerBase encode should return expected IDs"
        assert result_tokens == expected_tokens, "PreTrainedTokenizerBase convert_ids_to_tokens should return expected tokens"
        assert result_decoded == expected_decoded, "PreTrainedTokenizerBase decode should return expected text"
