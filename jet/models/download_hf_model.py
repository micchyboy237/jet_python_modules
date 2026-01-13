
from typing import List, Optional
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

from jet.logger import logger
from jet.models.config import MODELS_CACHE_DIR
from jet.models.download_hf_utils import ProgressBar, get_snapshot_settings, remove_download_cache
from jet.models.download_onnx_model import download_onnx_model
from jet.models.utils import resolve_model_value
from jet.models.onnx_model_checker import has_onnx_model_in_repo


def download_hf_model(
    repo_id: str,
    cache_dir: str = MODELS_CACHE_DIR,
    timeout: float = 300.0,
    clean_cache: bool = False,
) -> None:
    """
    Download a model from Hugging Face Hub.
    - First tries to download only safetensors + essential files.
    - If no safetensors file exists in the repo → automatically downloads the *.bin files instead.
    """
    try:
        model_path = resolve_model_value(repo_id)
    except ValueError:
        model_path = repo_id

    if clean_cache:
        remove_download_cache()

    # Resolve repo_id string once
    repo_id_str = str(model_path)

    try:
        settings = {
            **get_snapshot_settings(repo_id_str, cache_dir)
        }
        snapshot_download(**settings)
    except HfHubHTTPError as e:
        logger.error(f"Failed to download model from {repo_id_str}: {str(e)}")
        raise
    except TimeoutError:
        logger.error(f"Download timed out after {timeout} seconds for {repo_id_str}")
        raise


def download_hf_space(
    repo_id: str,
    cache_dir: str = MODELS_CACHE_DIR,
    allow_patterns: List[str] = ["ckpt/*"],
    ignore_patterns: Optional[List[str]] = None,
    clean_cache: bool = False,
    force_download: bool = False,
) -> None:
    """
    Download files from a Hugging Face Space repository.
    Uses snapshot_download with repo_type="space".

    Args:
        repo_id: Space repository ID (e.g., "litagin/Japanese-Ero-Voice-Classifier").
        cache_dir: Directory to cache downloaded files.
        allow_patterns: List of patterns to include (e.g., ["ckpt/*", "app.py"]).
        ignore_patterns: List of patterns to exclude.
        clean_cache: Whether to remove lock files and XET cache before starting.
        force_download: Force re-download even if files exist locally.
    """
    if clean_cache:
        remove_download_cache()

    logger.info(f"Starting download of Space: {repo_id}")

    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="space",  # Required for Spaces
            cache_dir=cache_dir,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            force_download=force_download,
            resume_download=True,
            local_dir_use_symlinks=False,
            max_workers=4,
            etag_timeout=20.0,
            tqdm_class=ProgressBar,
        )
        logger.info(f"Space {repo_id} downloaded successfully")
    except HfHubHTTPError as e:
        logger.error(f"HTTP error downloading Space {repo_id}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Failed to download Space {repo_id}: {str(e)}")
        raise


if __name__ == "__main__":
    repo_id = "litagin/anime_speech_emotion_classification"
    cache_dir = MODELS_CACHE_DIR
    clean_cache = False

    logger.info(f"Downloading files from repo id: {repo_id}...")

    try:
        download_hf_model(repo_id, clean_cache=clean_cache)

        if has_onnx_model_in_repo(repo_id):
            download_onnx_model(repo_id)

        # Do not clean cache after successful download unless explicitly requested
        # remove_download_cache()

        logger.info("Download completed")
    except Exception:
        logger.info(f"Downloading files from repo id (space): {repo_id}...")

        try:
            download_hf_space(repo_id, clean_cache=clean_cache)
        except Exception as e:
            logger.error(f"Download failed: {str(e)}")
            raise
