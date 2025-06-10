import pytest
import logging
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from jet.models.tokenizer.base import get_tokenizer

# Configure logging for tests
logging.basicConfig(level=logging.INFO)


class TestGetTokenizer:
    """Test suite for get_tokenizer function."""

    def test_load_tokenizer_remote_success(self, tmp_path, monkeypatch):
        """Test loading tokenizer from remote repository."""
        repo_id = "sentence-transformers/static-retrieval-mrl-en-v1"

        # Mock Tokenizer.from_pretrained to return a valid Tokenizer instance
        def mock_from_pretrained(repo_id):
            tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
            return tokenizer

        monkeypatch.setattr(Tokenizer, "from_pretrained", mock_from_pretrained)

        result = get_tokenizer(repo_id)
        expected = Tokenizer(WordPiece(unk_token="[UNK]"))

        assert isinstance(result, Tokenizer), "Expected a Tokenizer instance"
        assert isinstance(
            result.model, WordPiece), f"Expected WordPiece model, got {type(result.model)}"
        assert result.model.unk_token == expected.model.unk_token, f"Expected unk_token {expected.model.unk_token}, got {result.model.unk_token}"

    def test_load_tokenizer_from_nested_cache(self, tmp_path, monkeypatch):
        """Test loading tokenizer from nested directory in cache."""
        repo_id = "sentence-transformers/static-retrieval-mrl-en-v1"
        cache_dir = tmp_path / "cache"
        snapshot_dir = cache_dir / \
            f"models--{repo_id.replace('/', '--')}" / \
            "snapshots" / "f123456789"
        nested_dir = snapshot_dir / "nested" / "deep"
        tokenizer_path = nested_dir / "tokenizer.json"
        tokenizer_path.parent.mkdir(parents=True)

        # Create a dummy tokenizer.json with minimal valid content
        with open(tokenizer_path, "w") as f:
            f.write('{"type": "WordPiece", "unk_token": "[UNK]"}')

        # Mock Tokenizer.from_pretrained to fail
        def mock_from_pretrained(repo_id):
            raise ValueError("Remote fetch failed")

        # Mock Tokenizer.from_file to return a valid Tokenizer instance
        def mock_from_file(path):
            return Tokenizer(WordPiece(unk_token="[UNK]"))

        monkeypatch.setattr(Tokenizer, "from_pretrained", mock_from_pretrained)
        monkeypatch.setattr(Tokenizer, "from_file", mock_from_file)

        result = get_tokenizer(repo_id, local_cache_dir=str(cache_dir))
        expected = Tokenizer(WordPiece(unk_token="[UNK]"))

        assert isinstance(result, Tokenizer), "Expected a Tokenizer instance"
        assert isinstance(
            result.model, WordPiece), f"Expected WordPiece model, got {type(result.model)}"
        assert result.model.unk_token == expected.model.unk_token, f"Expected unk_token {expected.model.unk_token}, got {result.model.unk_token}"

    def test_load_tokenizer_from_symlink(self, tmp_path, monkeypatch):
        """Test loading tokenizer from a symlink in cache."""
        repo_id = "sentence-transformers/static-retrieval-mrl-en-v1"
        cache_dir = tmp_path / "cache"
        snapshot_dir = cache_dir / \
            f"models--{repo_id.replace('/', '--')}" / \
            "snapshots" / "f123456789"
        real_dir = snapshot_dir / "real"
        real_tokenizer_path = real_dir / "tokenizer.json"
        symlink_path = snapshot_dir / "link" / "tokenizer.json"

        # Create real tokenizer.json
        real_dir.mkdir(parents=True)
        with open(real_tokenizer_path, "w") as f:
            f.write('{"type": "WordPiece", "unk_token": "[UNK]"}')

        # Create symlink to real tokenizer.json
        symlink_path.parent.mkdir(parents=True)
        symlink_path.symlink_to(real_tokenizer_path)

        # Mock Tokenizer.from_pretrained to fail
        def mock_from_pretrained(repo_id):
            raise ValueError("Remote fetch failed")

        # Mock Tokenizer.from_file to return a valid Tokenizer instance
        def mock_from_file(path):
            return Tokenizer(WordPiece(unk_token="[UNK]"))

        monkeypatch.setattr(Tokenizer, "from_pretrained", mock_from_pretrained)
        monkeypatch.setattr(Tokenizer, "from_file", mock_from_file)

        result = get_tokenizer(repo_id, local_cache_dir=str(cache_dir))
        expected = Tokenizer(WordPiece(unk_token="[UNK]"))

        assert isinstance(result, Tokenizer), "Expected a Tokenizer instance"
        assert isinstance(
            result.model, WordPiece), f"Expected WordPiece model, got {type(result.model)}"
        assert result.model.unk_token == expected.model.unk_token, f"Expected unk_token {expected.model.unk_token}, got {result.model.unk_token}"

    def test_tokenizer_failure(self, tmp_path, monkeypatch):
        """Test failure when both remote and local loading fail."""
        repo_id = "sentence-transformers/static-retrieval-mrl-en-v1"
        cache_dir = tmp_path / "cache"
        snapshot_dir = cache_dir / \
            f"models--{repo_id.replace('/', '--')}" / \
            "snapshots" / "f123456789"
        snapshot_dir.mkdir(parents=True)

        # Mock remote failure
        def mock_from_pretrained(repo_id):
            raise ValueError("Remote fetch failed")

        # Mock local file loading failure
        def mock_from_file(path):
            raise FileNotFoundError("File not found")

        monkeypatch.setattr(Tokenizer, "from_pretrained", mock_from_pretrained)
        monkeypatch.setattr(Tokenizer, "from_file", mock_from_file)

        with pytest.raises(ValueError) as exc_info:
            get_tokenizer(repo_id, local_cache_dir=str(cache_dir))

        result = str(exc_info.value)
        expected = f"Could not load tokenizer for {repo_id} from remote or local cache."

        assert result == expected, f"Expected error message {expected}, got {result}"
