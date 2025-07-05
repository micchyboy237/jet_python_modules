import logging
from typing import List, Optional, Union
import os
from huggingface_hub import HfApi, list_repo_files
from jet.logger import logger
from jet.models.config import MODELS_CACHE_DIR
from jet.models.model_types import ModelType


def has_onnx_model_in_repo(repo_id: Union[str, ModelType], token: Optional[str] = None) -> bool:
    """
    Check if any ONNX model (standard model.onnx, model_*_arm64.onnx, or model*quantized*onnx) exists in a Hugging Face model repository.
    Checks the local cache first, then falls back to the remote repository if no local models are found.

    Args:
        repo_id (str): The ID of the repository (e.g., "sentence-transformers/all-MiniLM-L6-v2").
        token (Optional[str]): Hugging Face API token for private repositories.

    Returns:
        bool: True if a standard, ARM64, or quantized ONNX model is found, False otherwise.
    """
    from jet.models.utils import resolve_model_value

    try:
        repo_id = resolve_model_value(repo_id)
        logger.info(f"Checking for ONNX models in repository: {repo_id}")

        # Check local cache first
        local_onnx_paths = get_onnx_model_paths(
            repo_id, cache_dir=MODELS_CACHE_DIR, token=token)
        if local_onnx_paths:
            logger.info(
                f"ONNX model(s) found in local cache for {repo_id}: {local_onnx_paths}")
            return True

        # Fall back to remote repository check
        logger.info(
            f"No ONNX models found in local cache, checking remote repository: {repo_id}")
        api = HfApi()
        repo_files = api.list_repo_files(repo_id=repo_id, token=token)
        logger.debug(f"Files found in {repo_id}: {repo_files}")
        has_onnx = (
            "model.onnx" in repo_files or
            any(file.startswith("model_") and file.endswith("_arm64.onnx") for file in repo_files) or
            any("quantized" in file and file.endswith(".onnx")
                for file in repo_files)
        )
        logger.info(
            f"ONNX model (standard, ARM64, or quantized) found in {repo_id}: {has_onnx}")
        return has_onnx
    except Exception as e:
        logger.error(
            f"Error checking ONNX models in repository {repo_id}: {str(e)}")
        return False


def get_onnx_model_paths(repo_id: Union[str, ModelType], cache_dir: str = MODELS_CACHE_DIR, token: Optional[str] = None) -> List[str]:
    """
    Retrieve a list of ONNX model file paths (standard, ARM64, or quantized) in the local Hugging Face cache for a repository.

    Args:
        repo_id (str): The ID of the repository (e.g., "sentence-transformers/all-MiniLM-L6-v2").
        token (Optional[str]): Hugging Face API token (unused for local checks but included for consistency).

    Returns:
        List[str]: List of absolute ONNX model file paths found in the local cache.
    """
    try:
        logger.info(
            f"Retrieving local ONNX model paths for repository: {repo_id}")
        # Convert repo_id to cache folder name (e.g., "sentence-transformers/all-MiniLM-L6-v2" to cache folder)
        repo_folder_name = repo_id.replace("/", "--")
        repo_path = os.path.join(cache_dir, f"models--{repo_folder_name}")

        # Check if the repo exists in the local cache
        if not os.path.exists(repo_path):
            logger.warning(
                f"Repository {repo_id} not found in local cache at {repo_path}")
            return []

        # Walk through the repo directory to find ONNX files
        onnx_paths = []
        for root, _, files in os.walk(repo_path):
            for file in files:
                if (
                    file == "model.onnx" or
                    (file.startswith("model_") and file.endswith("_arm64.onnx")) or
                    ("quantized" in file and file.endswith(".onnx"))
                ):
                    full_path = os.path.join(root, file)
                    onnx_paths.append(full_path)

        logger.info(
            f"Found {len(onnx_paths)} ONNX model paths for {repo_id}: {onnx_paths}")
        return sorted(onnx_paths)
    except Exception as e:
        logger.error(
            f"Error retrieving local ONNX model paths for repository {repo_id}: {str(e)}")
        return []


def check_local_models_with_onnx() -> dict[str, bool]:
    """
    Check local models for ONNX model availability (standard, ARM64, or quantized).

    Returns:
        dict[str, bool]: Dictionary mapping model IDs to their ONNX availability.
    """
    from jet.models.utils import scan_local_hf_models

    local_models = scan_local_hf_models()
    results = {}
    for model in local_models:
        results[model] = has_onnx_model_in_repo(model)
    return results


# Example usage
if __name__ == "__main__":
    repo_id = "sentence-transformers/all-MiniLM-L6-v2"
    result = has_onnx_model_in_repo(repo_id)
    print(
        f"ONNX model (standard, ARM64, or quantized) exists in {repo_id}: {result}")

    onnx_paths = get_onnx_model_paths(repo_id)
    print(f"\nONNX model paths in {repo_id}:")
    for path in onnx_paths:
        print(f"- {path}")

    onnx_results = check_local_models_with_onnx()
    print("\nONNX Model Availability:")
    for model, has_onnx in onnx_results.items():
        print(f"{model}: {'yes' if has_onnx else 'no'}")
