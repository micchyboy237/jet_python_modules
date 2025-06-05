import os
from jet.models.model_types import ModelKey, ModelType, ModelValue
import requests
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional

from huggingface_hub import (
    CachedRepoInfo,
    scan_cache_dir,
    snapshot_download,
    hf_hub_download,
    HfApi
)

# Mapping of friendly names to Hugging Face model repo IDs
EMBED_MODELS = {
    "nomic-embed-text": "nomic-ai/nomic-embed-text-v1.5",
    "mxbai-embed-large": "mixedbread-ai/mxbai-embed-large-v1",
    "granite-embedding": "ibm-granite/granite-embedding-30m-english",
    "granite-embedding:278m": "ibm-granite/granite-embedding-278m-multilingual",
    "all-minilm:22m": "sentence-transformers/all-MiniLM-L6-v2",
    "all-minilm:33m": "sentence-transformers/all-MiniLM-L12-v2",
    "snowflake-arctic-embed:33m": "Snowflake/snowflake-arctic-embed-s",
    "snowflake-arctic-embed:137m": "Snowflake/snowflake-arctic-embed-m-long",
    "snowflake-arctic-embed": "Snowflake/snowflake-arctic-embed-l",
    "paraphrase-multilingual": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "bge-large": "BAAI/bge-large-en-v1.5",
}

hf_api = HfApi()


def download_huggingface_repo(
    repo_id: str,
    cache_dir: Optional[str] = None,
    max_workers: int = 4,
    use_symlinks: bool = False
) -> str:
    """
    Download a Hugging Face model repository snapshot.

    Args:
        repo_id (str): The Hugging Face model or dataset repository ID.
        cache_dir (Optional[str]): Local cache directory. Defaults to ~/.cache/huggingface/hub.
        max_workers (int): Number of download threads to use.
        use_symlinks (bool): Whether to use symlinks in the local_dir.

    Returns:
        str: Path to the downloaded snapshot.
    """
    resolved_cache_dir = (
        cache_dir or os.path.expanduser("~/.cache/huggingface/hub")
    )

    path = snapshot_download(
        repo_id=repo_id,
        cache_dir=resolved_cache_dir,
        local_dir_use_symlinks=use_symlinks,
        max_workers=max_workers
    )
    print(f"Model repository downloaded to: {path}")
    return path


def download_readme(model_id: str | ModelValue, model_name: ModelType, output_dir: Path, overwrite: bool = False) -> bool:
    """
    Download README.md for a Hugging Face model using API, fallback to web scraping if necessary.

    Args:
        model_id (str | ModelValue): Full HF model repo ID or model name.
        model_name (str): Local name for saved README.
        output_dir (Path): Directory to save README.
        overwrite (bool): Force overwrite if file exists.

    Returns:
        bool: True if successful, False otherwise.
    """
    from jet.models.utils import resolve_model_key, resolve_model_value

    model_key = resolve_model_key(model_name)
    readme_path = output_dir / f"{model_key}_README.md"

    if readme_path.exists() and not overwrite:
        print(f"README for {model_key} already exists, skipping...")
        return True

    # Try API download
    try:
        hf_hub_download(
            repo_id=model_id,
            filename="README.md",
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        downloaded_file = output_dir / "README.md"
        if downloaded_file.exists():
            downloaded_file.rename(readme_path)
        print(f"Downloaded README for {model_key} via API")
        return True
    except Exception as e:
        print(f"API failed for {model_key}: {e}")

    # Fallback to web scraping
    try:
        url = f"https://huggingface.co/{model_id}/raw/main/README.md"
        response = requests.get(url)
        if response.status_code == 200:
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            print(f"Downloaded README for {model_key} via web")
            return True
        else:
            print(
                f"Web scraping failed for {model_key}: HTTP {response.status_code}")
    except Exception as e:
        print(f"Web scraping failed for {model_key}: {e}")

    return False


def download_model_readmes(output_dir: str = "hf_readmes", overwrite: bool = False) -> None:
    """
    Download READMEs for all models in ALL_MODELS dictionary.

    Args:
        output_dir (str): Directory to store downloaded READMEs.
        overwrite (bool): Whether to overwrite existing files.
    """
    from .utils import resolve_model_key
    from .constants import MODEL_VALUES_LIST

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # local_models = ALL_MODELS
    model_values = scan_local_hf_models()
    filtered_model_values = [
        model_value for model_value in model_values if model_value]
    model_keys = [resolve_model_key(
        model_path) for model_path in model_values if model_path in MODEL_VALUES_LIST]
    for model_key, model_value in zip(model_keys, filtered_model_values):
        print(f"Processing {model_key} ({model_value})...")
        download_readme(model_value, model_key, output_dir_path, overwrite)

    print(f"All READMEs saved to {output_dir_path}")


def scan_local_hf_models() -> List[str]:
    hf_cache_info = scan_cache_dir()
    downloaded_models: FrozenSet[CachedRepoInfo] = hf_cache_info.repos
    local_models = [
        repo.repo_id for repo in downloaded_models
        if repo.repo_type == "model"
    ]
    return sorted(local_models)


def scan_local_hf_datasets() -> List[str]:
    hf_cache_info = scan_cache_dir()
    downloaded_models: FrozenSet[CachedRepoInfo] = hf_cache_info.repos
    local_models = [
        repo.repo_id for repo in downloaded_models
        if repo.repo_type == "dataset"
    ]
    return sorted(local_models)


def scan_local_hf_spaces() -> List[str]:
    hf_cache_info = scan_cache_dir()
    downloaded_models: FrozenSet[CachedRepoInfo] = hf_cache_info.repos
    local_models = [
        repo.repo_id for repo in downloaded_models
        if repo.repo_type == "space"
    ]
    return sorted(local_models)
