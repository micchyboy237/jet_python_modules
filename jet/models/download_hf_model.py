from pathlib import Path
import shutil
from typing import Union, Optional, Type
import os
import glob
import time
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError
import sys
from datetime import datetime

from jet.logger import logger
from jet.models.config import MODELS_CACHE_DIR, XET_CACHE_DIR
from jet.models.model_types import ModelType
from jet.models.download_onnx_model import download_onnx_model
from jet.models.utils import resolve_model_value
from jet.models.onnx_model_checker import has_onnx_model_in_repo


class ProgressBar:
    """Custom progress bar implementation mimicking tqdm behavior."""

    def __init__(self, total: Optional[int] = None, unit: str = "B", unit_scale: bool = True, desc: str = ""):
        self.total = total
        self.unit = unit
        self.unit_scale = unit_scale
        self.desc = desc
        self.current = 0
        self.start_time = datetime.now()
        self._last_update = 0
        self._width = 50  # Width of the progress bar

    def update(self, n: int) -> None:
        """Update progress bar by n units."""
        self.current += n
        self._refresh()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close the progress bar and print newline."""
        print("", flush=True)

    def _refresh(self):
        """Refresh the progress bar display."""
        if not sys.stdout.isatty():
            return

        if self.total is None:
            # Simple counter for unknown total
            print(f"\r{self.desc}: {self._format_size(self.current)}",
                  end="", flush=True)
            return

        # Calculate progress
        progress = min(self.current / self.total, 1.0)
        filled = int(self._width * progress)
        bar = "█" * filled + "-" * (self._width - filled)

        # Format size and speed
        size_str = self._format_size(self.current)
        speed = self.current / \
            max((datetime.now() - self.start_time).total_seconds(), 0.001)
        speed_str = self._format_size(speed) + "/s"

        # Update display
        percent = progress * 100
        print(
            f"\r{self.desc}: |{bar}| {percent:.1f}% {size_str} {speed_str}", end="", flush=True)

    def _format_size(self, size: float) -> str:
        """Format size with appropriate units."""
        if not self.unit_scale:
            return f"{size:.1f}{self.unit}"

        for unit in ["", "K", "M", "G", "T"]:
            if size < 1000:
                return f"{size:.1f}{unit}{self.unit}"
            size /= 1000
        return f"{size:.1f}P{self.unit}"


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
                "*.onnx",
                "onnx/*.onnx",
                "onnx/*/*.onnx",
                "openvino/*",
            ],
            allow_patterns=None,
            force_download=False,
            etag_timeout=20.0,
            local_dir_use_symlinks="auto",
            max_workers=4,
            resume_download=True,
            tqdm_class=ProgressBar,
        )
    except HfHubHTTPError as e:
        logger.error(f"Failed to download model from {model_path}: {str(e)}")
        raise
    except TimeoutError:
        logger.error(
            f"Download timed out after {timeout} seconds for {model_path}")
        raise


if __name__ == "__main__":
    repo_id = "google/embeddinggemma-300m"
    cache_dir = MODELS_CACHE_DIR

    logger.info(f"Downloading files from repo id: {repo_id}...")

    try:
        logger.info(f"Removing lock files from cache directory: {cache_dir}")
        download_hf_model(repo_id)

        if has_onnx_model_in_repo(repo_id):
            download_onnx_model(repo_id)

        remove_download_cache()

        logger.info("Download completed")
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        raise
