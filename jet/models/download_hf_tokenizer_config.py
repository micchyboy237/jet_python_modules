from huggingface_hub import snapshot_download
import os

from jet.models.config import MODELS_CACHE_DIR


def download_tokenizer_files(repo_id, cache_dir):
    """Download only tokenizer-related files from a Hugging Face repository."""
    tokenizer_files = snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        allow_patterns=["tokenizer.json",
                        "tokenizer_config.json", "vocab.json", "merges.txt"],
        local_dir_use_symlinks=False,
        max_workers=4,
        force_download=True
    )
    return tokenizer_files


# Configuration
repo_id = "sentence-transformers/static-retrieval-mrl-en-v1"
cache_dir = MODELS_CACHE_DIR

# Download tokenizer files
downloaded_path = download_tokenizer_files(repo_id, cache_dir)
print(f"Tokenizer files downloaded to: {downloaded_path}")
