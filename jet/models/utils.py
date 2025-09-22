import tempfile
from huggingface_hub import (
    CachedRepoInfo,
    scan_cache_dir,
    snapshot_download,
    hf_hub_download,
    HfApi
)
from typing import Any, Dict, FrozenSet, List, Optional
from pathlib import Path
import requests
from jet.models.config import MODELS_CACHE_DIR
import os
from typing import Dict, List, Tuple, Optional, TypedDict, Union
from jet.models.onnx_model_checker import has_onnx_model_in_repo
from jet.transformers.formatters import format_json
from jet.utils.object import max_getattr
from jet.logger import logger
from transformers import AutoConfig
from jet.models.model_types import ModelKey, ModelType, ModelValue
from jet.models.constants import ALL_MODEL_VALUES, ALL_MODELS, ALL_MODELS_REVERSED, AVAILABLE_EMBED_MODELS, MODEL_CONTEXTS, MODEL_EMBEDDING_TOKENS, MODEL_VALUES_LIST


def resolve_model_key(model: ModelType) -> ModelKey:
    """
    Retrieves the model key (short name) for a given model key or path.

    Args:
        model: A model key (short name) or full model path.

    Returns:
        The corresponding model key (short name).

    Raises:
        ValueError: If the model key or path is not recognized.
    """
    if model in ALL_MODELS:
        return model
    elif model in ALL_MODEL_VALUES:
        return ALL_MODELS_REVERSED[model]
    for key, value in ALL_MODELS.items():
        if value == model:
            return key
    return model


def resolve_model_value(model: ModelType) -> ModelValue:
    """
    Retrieves the model value (full path) for a given model key or path.

    Args:
        model: A model key (short name) or full model path.

    Returns:
        The corresponding model value (full path).

    Raises:
        ValueError: If the model key or path is not recognized.
    """
    if model in ALL_MODELS:
        return ALL_MODELS[model]
    # if model in ALL_MODELS.values() or "/" in model:
    #     return model
    # raise ValueError(
    #     f"Invalid model: {model}. Must be one of: "
    #     f"{list(ALL_MODELS.keys()) + list(ALL_MODELS.values())}"
    # )
    return model


def get_model_limits(model_id: Union[str, 'ModelValue']) -> Tuple[Optional[int], Optional[int]]:
    """
    Get the maximum context length and embedding size for a given model.

    Args:
        model_id: The model identifier or ModelValue object.

    Returns:
        Tuple containing (max_context, max_embeddings).
    """
    try:
        model_path = resolve_model_value(model_id)
        # Check if model exists remotely
        try:
            HfApi().model_info(model_path)
            logger.info(f"Model {model_path} found on Hugging Face Hub")
        except Exception as e:
            logger.error(
                f"Model {model_path} not found on Hugging Face Hub: {str(e)}")
            return None, None

        # Try to load config from cache or remote
        try:
            config = AutoConfig.from_pretrained(
                model_path,
                trust_remote_code=True,
                # Set to True if using private repos
                token=os.getenv("HF_TOKEN"),
                cache_dir=None,  # Use default cache directory
                force_download=False,  # Use cached if available
                resume_download=False
            )
        except Exception as e:
            logger.error(f"Failed to load config for {model_path}: {str(e)}")
            return None, None

        # Helper function to safely get attributes
        def max_getattr(obj, attr, default=None):
            return getattr(obj, attr, default) if hasattr(obj, attr) else default

        max_context = max_getattr(config, 'max_position_embeddings', None)
        max_embeddings = max_getattr(config, 'hidden_size', None)

        if max_context is None and max_embeddings is None:
            logger.warning(f"No model limits found for {model_path}")

        return max_context, max_embeddings

    except Exception as e:
        logger.error(
            f"Unexpected error while processing {model_path}: {str(e)}")
        return None, None


def generate_short_name(model_id: str) -> Optional[str]:
    """
    Extract the final segment from a model ID or path and convert it to lowercase.

    Args:
        model_id (str): Model identifier or path (e.g., "mlx-community/Dolphin3.0-Llama3.1-8B-4bit")

    Returns:
        Optional[str]: Lowercase short name (e.g., "dolphin3.0-llama3.1-8b-4bit") or None if input is empty
    """
    try:
        if not model_id:
            logger.warning("Empty model_id provided")
            return None

        # Get the final path component and convert to lowercase
        short_name = os.path.basename(model_id).lower()
        logger.debug(f"Processed short name: {short_name}")
        return short_name
    except Exception as e:
        logger.error(f"Error processing model_id ({model_id}): {str(e)}")
        return None


class ModelInfoDict(TypedDict):
    models: Dict[ModelKey, ModelValue]
    contexts: Dict[ModelKey, int]
    embeddings: Dict[ModelKey, int]
    has_onnx: Dict[ModelKey, bool]
    missing: List[str]


def get_model_info() -> ModelInfoDict:
    model_info: ModelInfoDict = {
        "models": {},
        "contexts": {},
        "embeddings": {},
        "has_onnx": {},
        "missing": []
    }
    model_paths = scan_local_hf_models()
    for model_path in model_paths:
        try:
            if model_path not in MODEL_VALUES_LIST:
                short_name = generate_short_name(model_path)
            else:
                short_name = resolve_model_key(model_path)
            max_contexts, max_embeddings = get_model_limits(model_path)
            if not max_contexts:
                raise ValueError(
                    f"Missing 'max_position_embeddings' from {model_path} config")
            elif not max_embeddings:
                raise ValueError(
                    f"Missing 'hidden_size' from {model_path} config")

            print(
                f"{short_name}: max_contexts={max_contexts}, max_embeddings={max_embeddings}")

            model_info["models"][short_name] = model_path
            model_info["contexts"][short_name] = max_contexts
            model_info["embeddings"][short_name] = max_embeddings
            model_info["has_onnx"][short_name] = has_onnx_model_in_repo(
                model_path)

        except Exception as e:
            logger.error(
                f"Failed to get config for {short_name}: {str(e)[:100]}", exc_info=True)
            model_info["missing"].append(model_path)
            continue

    return model_info


def resolve_model(model_name: ModelType) -> ModelType:
    """
    Resolves a model name or path against available models.

    Args:
        model_name: A short key or full model path.

    Returns:
        The resolved full model path.

    Raises:
        ValueError: If the model name/path is not recognized.
    """
    if model_name in ALL_MODELS:
        return ALL_MODELS[model_name]
    # if model_name in ALL_MODELS.values() or "/" in model_name:
    #     return model_name
    # else:
    #     raise ValueError(
    #         f"Invalid model: {model_name}. Must be one of: "
    #         f"{list(ALL_MODELS.keys()) + list(ALL_MODELS.values())}"
    #     )
    return model_name


def get_embedding_size(model: ModelType) -> int:
    """
    Returns the embedding size (hidden dimension) for the given model key or full model path.

    Args:
        model: A model key or model path.

    Returns:
        The embedding size (hidden dimension).

    Raises:
        ValueError: If the model is not recognized or missing an embedding size.
    """
    model_key = resolve_model_key(model)
    if model_key not in MODEL_EMBEDDING_TOKENS:
        error_msg = f"Missing embedding size for model: {model_key}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    return MODEL_EMBEDDING_TOKENS[model_key]


def get_context_size(model: ModelType) -> int:
    """
    Returns the context size (hidden dimension) for the given model key or full model path.

    Args:
        model: A model key or model path.

    Returns:
        The maximum context size.

    Raises:
        ValueError: If the model is not recognized or missing an context size.
    """
    model_key = resolve_model_key(model)
    if model_key not in MODEL_CONTEXTS:
        error_msg = f"Missing context size for model: {model_key}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    return MODEL_CONTEXTS[model_key]


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
        cache_dir or MODELS_CACHE_DIR
    )

    path = snapshot_download(
        repo_id=repo_id,
        cache_dir=resolved_cache_dir,
        local_dir_use_symlinks=use_symlinks,
        max_workers=max_workers
    )
    print(f"Model repository downloaded to: {path}")
    return path


def download_readme(model_id: ModelType, output_dir: Union[str, Path], overwrite: bool = False, extract_code: bool = True) -> bool:
    """
    Download README.md for a Hugging Face model using API, fallback to web scraping if necessary.
    Optionally extract code blocks from the downloaded README.
    Args:
        model_id (str | ModelValue): Full HF model repo ID or model name.
        output_dir (str | Path): Directory to save README.
        overwrite (bool): Force overwrite if file exists.
        extract_code (bool): Whether to extract code blocks from the README.
    Returns:
        bool: True if download is successful, False otherwise.
    """
    from jet.models.utils import resolve_model_key
    from jet.models.extract_hf_readme_code import extract_code_from_hf_readmes
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        model_key = resolve_model_key(model_id)
        # Sanitize model_key to replace invalid characters for file/directory names
        sanitized_model_key = model_key.replace("/", "_").replace("\\", "_")
    except ValueError as e:
        logger.warning(
            f"Could not resolve model key for model_name='{model_id}': {e}")
        return False
    readme_path = output_dir / f"{sanitized_model_key}_README.md"
    if readme_path.exists() and not overwrite:
        print(f"README for {sanitized_model_key} already exists, skipping...")
        if extract_code:
            code_output_dir = output_dir / "code"
            extract_code_from_hf_readmes(
                str(output_dir), str(code_output_dir), model_id, include_text=True)
        return True
    try:
        with tempfile.TemporaryDirectory(prefix=f"tmp_{sanitized_model_key}_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            hf_hub_download(
                repo_id=model_id,
                filename="README.md",
                local_dir=tmp_path,
                local_dir_use_symlinks=False
            )
            downloaded_file = tmp_path / "README.md"
            if downloaded_file.exists():
                downloaded_file.rename(readme_path)
                logger.success(
                    f"Downloaded README for {sanitized_model_key} via API: {str(readme_path)}")
                if extract_code:
                    code_output_dir = output_dir / "code"
                    extract_code_from_hf_readmes(
                        str(output_dir), str(code_output_dir), model_id, include_text=True)
                return True
    except Exception as e:
        logger.error(f"API failed for {sanitized_model_key}: {e}")
    try:
        url = f"https://huggingface.co/{model_id}/raw/main/README.md"
        response = requests.get(url)
        if response.status_code == 200:
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            print(f"Downloaded README for {sanitized_model_key} via web")
            if extract_code:
                code_output_dir = output_dir / "code"
                extract_code_from_hf_readmes(
                    str(output_dir), str(code_output_dir), model_id, include_text=True)
            return True
        else:
            print(
                f"Web scraping failed for {sanitized_model_key}: HTTP {response.status_code}")
    except Exception as e:
        print(f"Web scraping failed for {sanitized_model_key}: {e}")
    return False


def download_model_readmes(output_dir: str = "hf_readmes", overwrite: bool = False, extract_code: bool = True) -> None:
    """
    Download READMEs for all models in MODEL_VALUES_LIST dictionary and optionally extract code blocks.

    Args:
        output_dir (str): Directory to store downloaded READMEs.
        overwrite (bool): Whether to overwrite existing files.
        extract_code (bool): Whether to extract code blocks from downloaded READMEs.
    """
    from .utils import resolve_model_key
    from .constants import MODEL_VALUES_LIST

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    model_values = scan_local_hf_models()
    filtered_model_values = [
        model_value for model_value in model_values if model_value]
    for model_value in filtered_model_values:
        print(f"Processing ({model_value})...")
        download_readme(model_value, output_dir_path, overwrite, extract_code)

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


def get_repo_files(repo_id: str, token: Optional[str] = None) -> List[str]:
    """
    Retrieve list of files in a HuggingFace repository.

    Args:
        repo_id: Repository ID (e.g., 'sentence-transformers/all-MiniLM-L6-v2')
        token: Optional HuggingFace API token

    Returns:
        List of file paths in the repository
    """
    try:
        logger.info(f"Checking for files in repository: {repo_id}")
        api = HfApi()
        repo_files = api.list_repo_files(repo_id=repo_id, token=token)
        logger.debug(f"Files found in {repo_id}: {repo_files}")
        return repo_files
    except Exception as e:
        logger.error(f"Error checking files in repository {repo_id}: {str(e)}")
        return []


def get_local_repo_dir(repo_id: ModelType, filename: Optional[str] = None) -> Optional[str]:
    """
    Retrieve the local directory path for a Hugging Face repository.

    Args:
        repo_id (ModelType): The Hugging Face repository ID (e.g., 'nomic-ai/nomic-embed-text-v1.5').
        filename (Optional[str]): Optional filename to search for within the repository cache.

    Returns:
        Optional[str]: The local directory path if the repository exists locally, or the file path if filename is specified, None otherwise.
    """
    try:
        if filename:
            cache_dir = Path(repo_id)
            files = list(cache_dir.rglob(filename))
            logger.debug(
                f"Found {len(files)} {filename} files in {cache_dir}")
            if files:
                return str(files[0])

        repo_id = resolve_model_value(repo_id)
        logger.info(f"Scanning for local repository: {repo_id}")
        hf_cache_info = scan_cache_dir()

        for repo in hf_cache_info.repos:
            if repo.repo_id == repo_id and repo.repo_type == "model":
                local_dir = repo.repo_path
                logger.debug(
                    f"Found local directory for {repo_id}: {local_dir}")

                return str(local_dir)

        logger.warning(f"No local directory found for repository: {repo_id}")
        return None
    except Exception as e:
        logger.error(
            f"Error scanning for local repository {repo_id}: {str(e)}")
        return None
