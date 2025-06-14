import logging
from typing import List, Optional
from huggingface_hub import HfApi, list_repo_files

from jet.models.base import scan_local_hf_models

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def has_onnx_model_in_repo(repo_id: str, token: Optional[str] = None) -> bool:
    """
    Check if an ONNX model exists in a Hugging Face model repository.

    Args:
        repo_id (str): The ID of the repository (e.g., "sentence-transformers/all-MiniLM-L6-v2").
        token (Optional[str]): Hugging Face API token for private repositories.

    Returns:
        bool: True if an ONNX model is found, False otherwise.
    """
    try:
        logger.info(f"Checking for ONNX models in repository: {repo_id}")
        # Initialize Hugging Face API client
        api = HfApi()
        # List all files in the repository
        repo_files = api.list_repo_files(repo_id=repo_id, token=token)
        logger.debug(f"Files found in {repo_id}: {repo_files}")
        # Check for files with .onnx extension
        has_onnx = any(file.endswith(".onnx") for file in repo_files)
        logger.info(f"ONNX model found in {repo_id}: {has_onnx}")
        return has_onnx
    except Exception as e:
        logger.error(f"Error checking repository {repo_id}: {str(e)}")
        return False


def check_local_models_with_onnx() -> dict[str, bool]:
    local_models = scan_local_hf_models()
    has_onnx_list = []
    for model in local_models:
        has_onnx_list.append(has_onnx_model_in_repo(model))
    return {model: has_onnx for model, has_onnx in zip(local_models, has_onnx_list)}


# Example usage
if __name__ == "__main__":
    repo_id = "sentence-transformers/all-MiniLM-L6-v2"
    result = has_onnx_model_in_repo(repo_id)
    print(f"ONNX model exists in {repo_id}: {result}")

    onnx_results = check_local_models_with_onnx()
    print("\nONNX Model Availability:")
    for model, has_onnx in onnx_results.items():
        print(f"{model}: {'✓' if has_onnx else '✗'}")
