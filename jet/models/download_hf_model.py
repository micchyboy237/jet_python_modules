from pathlib import Path
import shutil
from typing import Union
import os
import glob
import time
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

from jet.logger import logger
from jet.models.config import MODELS_CACHE_DIR, XET_CACHE_DIR
from jet.models.model_types import ModelType
from jet.models.download_onnx_model import download_onnx_model
from jet.models.utils import resolve_model_value


def remove_cache_locks(cache_dir: str = MODELS_CACHE_DIR, max_attempts: int = 5, wait_interval: float = 0.1) -> None:
    """
    Remove all lock files from the specified cache directory with retries.

    Args:
        cache_dir (str): Path to the cache directory
        max_attempts (int): Maximum number of attempts to remove lock files
        wait_interval (float): Wait time between retry attempts in seconds
    """
    try:
        lock_pattern = os.path.join(cache_dir, "**", "*.lock")
        for attempt in range(1, max_attempts + 1):
            lock_files = glob.glob(lock_pattern, recursive=True)
            if not lock_files:
                logger.debug("No lock files found in cache directory")
                return
            for lock_file in lock_files:
                try:
                    os.remove(lock_file)
                    logger.debug(f"Removed lock file: {lock_file}")
                except OSError as e:
                    logger.warning(
                        f"Attempt {attempt}: Failed to remove lock file {lock_file}: {str(e)}")
            if lock_files and attempt < max_attempts:
                time.sleep(wait_interval)
        if glob.glob(lock_pattern, recursive=True):
            logger.warning(
                "Some lock files could not be removed after maximum attempts")
    except Exception as e:
        logger.error(f"Error while removing lock files: {str(e)}")
        raise


def remove_download_cache() -> None:
    shutil.rmtree(XET_CACHE_DIR, ignore_errors=True)

    remove_cache_locks()


def download_hf_model(repo_id: Union[str, ModelType], cache_dir: str = MODELS_CACHE_DIR, timeout: float = 300.0):
    """
    Download a model from Hugging Face Hub with optimized lock handling and timeout.

    Args:
        repo_id (Union[str, ModelType]): Repository ID or model type
        cache_dir (str): Directory to cache the downloaded model
        timeout (float): Maximum time to wait for download in seconds
    """
    try:
        model_path = resolve_model_value(repo_id)
    except ValueError:
        model_path = repo_id

    remove_download_cache()

    try:
        snapshot_download(
            repo_id=model_path,
            cache_dir=cache_dir,
            ignore_patterns=[
                "*.bin",
                "*.h5",
                "onnx/model*.onnx",  # Simplified pattern to cover all ONNX variants
                "openvino/*",
            ],
            local_dir_use_symlinks=False,
            force_download=True,
            etag_timeout=20.0,  # Set timeout for ETag fetching
        )
    except HfHubHTTPError as e:
        logger.error(f"Failed to download model from {model_path}: {str(e)}")
        raise
    except TimeoutError:
        logger.error(
            f"Download timed out after {timeout} seconds for {model_path}")
        raise

    download_onnx_model(repo_id)

    remove_download_cache()


if __name__ == "__main__":
    repo_id = "tomaarsen/static-retrieval-mrl-en-v1"
    cache_dir = MODELS_CACHE_DIR

    logger.info(f"Downloading files from repo id: {repo_id}...")

    try:
        logger.info(f"Removing lock files from cache directory: {cache_dir}")
        download_hf_model(repo_id)

        logger.info("Download completed")
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        raise
