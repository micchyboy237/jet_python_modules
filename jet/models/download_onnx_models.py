import logging
from typing import Optional
from huggingface_hub import snapshot_download, get_hf_file_metadata, hf_hub_url
from tqdm import tqdm
from jet.models.base import get_repo_files, scan_local_hf_models, EMBED_MODELS

logger = logging.getLogger(__name__)


def download_onnx_models(cache_dir: str = "/Users/jethroestrada/.cache/huggingface/hub", token: Optional[str] = None) -> None:
    """
    Iterate over local models, check for ARM64 ONNX models first, then standard ONNX models,
    and download only one based on priority (ARM64 preferred) if file size is <= 200MB.
    Display progress with a progress bar and log a summary of results.

    Args:
        cache_dir: Directory to store downloaded models
        token: Optional HuggingFace API token for private repos
    """
    logger.info("Starting ONNX model download process")
    local_models = scan_local_hf_models()
    total_models = len(local_models)
    logger.info(f"Found {total_models} local models to process")

    downloaded = 0
    skipped = 0
    failed = 0
    size_exceeded = 0

    # Iterate with progress bar
    for repo_id in tqdm(local_models, desc="Processing models", unit="model"):
        # Skip models not in EMBED_MODELS
        if repo_id not in EMBED_MODELS.values():
            logger.debug(f"Skipping {repo_id} as it's not in EMBED_MODELS")
            skipped += 1
            continue

        logger.info(f"Processing repository: {repo_id}")

        # Get repository files
        try:
            repo_files = get_repo_files(repo_id, token)
            if not repo_files:
                logger.info(f"No files found for {repo_id}, skipping download")
                skipped += 1
                continue

            # Define patterns in order of priority
            patterns = [
                "onnx/model_qint8_arm64.onnx",  # ARM64 model
                "onnx/model.onnx"              # Standard model
            ]

            selected_pattern = None
            for pattern in patterns:
                if any(pattern in file for file in repo_files):
                    # Check file size
                    file_url = hf_hub_url(repo_id=repo_id, filename=pattern)
                    try:
                        metadata = get_hf_file_metadata(file_url, token=token)
                        file_size_mb = metadata.size / \
                            (1024 * 1024)  # Convert bytes to MB
                        if file_size_mb > 200:
                            logger.info(
                                f"Skipping {pattern} for {repo_id}: File size {file_size_mb:.2f}MB exceeds 200MB limit")
                            size_exceeded += 1
                            continue
                        selected_pattern = pattern
                        break
                    except Exception as e:
                        logger.warning(
                            f"Could not retrieve metadata for {pattern} in {repo_id}: {str(e)}")
                        continue

            if selected_pattern:
                logger.info(
                    f"Downloading {selected_pattern} for {repo_id} (size: {file_size_mb:.2f}MB)")
                snapshot_download(
                    repo_id=repo_id,
                    cache_dir=cache_dir,
                    allow_patterns=[selected_pattern],
                    local_dir_use_symlinks=False,
                    force_download=True,
                    token=token
                )
                logger.info(
                    f"Successfully downloaded {selected_pattern} for {repo_id}")
                downloaded += 1
            else:
                logger.info(f"No suitable ONNX files found for {repo_id}")
                skipped += 1

        except Exception as e:
            logger.error(f"Failed to process {repo_id}: {str(e)}")
            failed += 1
            continue

    # Log summary
    logger.info(
        f"Download process completed. Summary:\n"
        f"- Total models processed: {total_models}\n"
        f"- Models downloaded: {downloaded}\n"
        f"- Models skipped: {skipped}\n"
        f"- Models skipped due to size (>200MB): {size_exceeded}\n"
        f"- Models failed: {failed}"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    download_onnx_models()
