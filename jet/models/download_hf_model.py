from huggingface_hub import (
    DatasetInfo,
    HfApi,
    snapshot_download,
)
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError
from jet.logger import logger
from jet.models.config import MODELS_CACHE_DIR
from jet.models.download_hf_utils import (
    ProgressBar,
    get_snapshot_settings,
    remove_download_cache,
)
from jet.models.download_onnx_model import download_onnx_model
from jet.models.onnx_model_checker import has_onnx_model_in_repo
from jet.models.utils import resolve_model_value


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
        settings = {**get_snapshot_settings(repo_id_str, cache_dir)}
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
    allow_patterns: list[str] = ["ckpt/*"],
    ignore_patterns: list[str] | None = None,
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


def download_dataset(
    repo_id: str,
    cache_dir: str = MODELS_CACHE_DIR,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
    revision: str | None = None,
    token: str | None = None,
    clean_cache: bool = False,
    force_download: bool = False,
    resume_download: bool = True,
    max_workers: int = 8,
    timeout: float = 30.0,
) -> str:
    """
    Download a full dataset (or selected files) from Hugging Face Hub.
    """
    if clean_cache:
        remove_download_cache()

    logger.info(f"Attempting to download as dataset: {repo_id}")

    api = HfApi()

    # Step 1: Verify it's actually a dataset
    try:
        repo_info: DatasetInfo = api.dataset_info(
            repo_id=repo_id,
            token=token,
            timeout=timeout,
            # files_metadata=False,   # usually not needed here
            # revision=revision,
        )
        logger.debug("Repository confirmed as dataset")
    except RepositoryNotFoundError:
        raise ValueError(
            f"'{repo_id}' is not a dataset repository (or does not exist / no access)."
        ) from None
    except HfHubHTTPError as e:
        if e.response is not None and e.response.status_code in (401, 403):
            raise ValueError(
                f"Access denied to dataset '{repo_id}'. "
                "Probably gated/private — provide a valid Hugging Face token."
            ) from e
        raise
    except Exception as e:
        raise RuntimeError(f"Unexpected error checking dataset info: {str(e)}") from e

    # Step 2: Proceed with download (now confident it's a dataset)
    try:
        local_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            cache_dir=cache_dir,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            revision=revision,
            token=token,
            force_download=force_download,
            resume_download=resume_download,
            local_dir_use_symlinks=False,
            max_workers=max_workers,
            etag_timeout=timeout,
            tqdm_class=ProgressBar,
        )
        logger.info(f"Dataset downloaded successfully → {local_path}")
        return local_path

    except Exception as e:
        logger.error(f"Dataset snapshot download failed: {str(e)}")
        raise


# ──────────────────────────────────────────────
#          Example usage
# ──────────────────────────────────────────────

if __name__ == "__main__":
    repo_id = "m-ric/huggingface_doc"
    cache_dir = MODELS_CACHE_DIR
    clean_cache = False

    logger.info(f"Attempting to download repo: {repo_id}...")

    success = False

    # 1. Try as regular model (most common case)
    try:
        logger.info("Trying to download as model...")
        download_hf_model(repo_id, cache_dir=cache_dir, clean_cache=clean_cache)

        # Optional: check for ONNX variant
        if has_onnx_model_in_repo(repo_id):
            logger.info("Found ONNX files — downloading ONNX model...")
            download_onnx_model(repo_id)

        success = True
        logger.info("Download completed successfully (model)")

    except Exception as e1:
        logger.warning(f"Model download failed: {str(e1)}")

        # 2. Try as Space
        try:
            logger.info("Trying to download as Space...")
            download_hf_space(
                repo_id,
                cache_dir=cache_dir,
                clean_cache=clean_cache,
                # You can customize allow_patterns here if needed for your use-case
                # allow_patterns=["*.py", "*.json", "ckpt/*", "data/*"],
            )
            success = True
            logger.info("Download completed successfully (space)")

        except Exception as e2:
            logger.warning(f"Space download failed: {str(e2)}")

            # 3. Try as Dataset (third fallback)
            try:
                logger.info("Trying to download as Dataset...")
                local_path = download_dataset(
                    repo_id,
                    cache_dir=cache_dir,
                    clean_cache=clean_cache,
                    # Optional: common dataset patterns — adjust as needed
                    # allow_patterns=["data/*", "*.parquet", "*.jsonl", "*.json", "*.csv"],
                    # ignore_patterns=["*.gitattributes", "README.md"],
                )
                success = True
                logger.info(f"Download completed successfully (dataset) → {local_path}")

            except Exception as e3:
                logger.error(f"Dataset download also failed: {str(e3)}")

    if not success:
        logger.error(
            "All download attempts failed.\n"
            "Tried: model → space → dataset.\n"
            "Please check:\n"
            "  - whether the repo exists\n"
            "  - repo type (model/space/dataset)\n"
            "  - access rights (gated repo?)\n"
            "  - correct repo_id spelling"
        )
        raise RuntimeError(f"Failed to download repository: {repo_id}")

    # Optional final cleanup (only if explicitly requested via flag)
    # if clean_cache:
    #     remove_download_cache()
