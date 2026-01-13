from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError
import shutil
import os
import glob
import time
import sys
import threading

from datetime import datetime

from jet.logger import logger
from jet.models.config import MODELS_CACHE_DIR, XET_CACHE_DIR


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

def has_safetensors_in_repo(repo_id: str) -> bool:
    """
    Quick check if the repository contains any safetensors file.
    Uses the lightweight `list_repo_files` API – no download required.
    """
    from huggingface_hub import list_repo_files

    try:
        files = list_repo_files(repo_id=repo_id, repo_type="model")
        return any(f.endswith(".safetensors") for f in files)
    except Exception as e:
        logger.warning(f"Could not list files for {repo_id}: {e}")
        return False

def get_snapshot_settings(repo_id: str, cache_dir: str = MODELS_CACHE_DIR):
    """
    Return a dict of snapshot_download settings for a HF repo, matching defaults from download_hf_model.py.

    Default values from context:
        allow_patterns: None
        ignore_patterns: calculated above
        force_download: False
        etag_timeout: 20.0
        local_dir_use_symlinks: "auto"
        max_workers: 4
        resume_download: True
        tqdm_class: ProgressBar
        cache_dir: cache_dir (propagated in download_hf_model.py)
        repo_id: repo_id (passed separately)
    """
    # Determine which patterns to allow
    has_st = has_safetensors_in_repo(repo_id)

    if has_st:
        # Prefer safetensors – ignore the old .bin weights
        allow_patterns = None
        ignore_patterns = [
            "*.bin",
            "*.h5",
            "*.msgpack",
            "*.onnx",
            "onnx/*.onnx",
            "onnx/*/*.onnx",
            "openvino/*",
        ]
        logger.info(f"Repository {repo_id} contains safetensors → ignoring *.bin files")
    else:
        # No safetensors → download the legacy .bin weights (and everything else except ONNX/OpenVINO)
        allow_patterns = None
        ignore_patterns = [
            "*.onnx",
            "onnx/*.onnx",
            "onnx/*/*.onnx",
            "openvino/*",
        ]
        logger.info(f"No safetensors found in {repo_id} → downloading *.bin weights")

    return {
        "repo_id": repo_id,
        "cache_dir": cache_dir,
        "allow_patterns": allow_patterns,
        "ignore_patterns": ignore_patterns,
        "force_download": False,
        "etag_timeout": 20.0,
        "local_dir_use_symlinks": "auto",
        "max_workers": 4,
        "resume_download": True,
        "tqdm_class": ProgressBar,
    }

def download_single_repo(
    repo_id: str,
    cache_dir: str = MODELS_CACHE_DIR,
    max_workers: int = 4,
    **snapshot_kwargs,
) -> str:
    """
    Download a single HF repository with error handling.
    Returns the local snapshot path on success.
    """
    try:
        local_path = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            max_workers=max_workers,
            resume_download=True,
            local_dir_use_symlinks="auto",
            tqdm_class=ProgressBar,  # your custom one, or use tqdm directly
            **snapshot_kwargs,
        )
        logger.info(f"Successfully downloaded {repo_id} → {local_path}")
        return local_path
    except HfHubHTTPError as e:
        logger.error(f"HTTP error downloading {repo_id}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to download {repo_id}: {e}")
        raise

def download_multiple_repos_parallel(
    repo_ids: List[str],
    cache_dir: str = MODELS_CACHE_DIR,
    max_concurrent: int = 4,
    max_workers_per_download: int = 4,
    **snapshot_kwargs,
) -> List[str]:
    """
    Download multiple Hugging Face repositories in parallel.
    
    Args:
        repo_ids: List of repository IDs to download.
        max_concurrent: Maximum number of repositories to download simultaneously.
        max_workers_per_download: Threads per individual download (leverages HF's internal parallelism).
        snapshot_kwargs: Additional kwargs passed to snapshot_download (e.g., allow_patterns, ignore_patterns).
    
    Returns:
        List of local snapshot paths in completion order.
    """
    results = []
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        future_to_repo = {
            executor.submit(
                download_single_repo,
                repo_id=repo_id,
                cache_dir=cache_dir,
                max_workers=max_workers_per_download,
                **snapshot_kwargs,
            ): repo_id
            for repo_id in repo_ids
        }

        for future in as_completed(future_to_repo):
            repo_id = future_to_repo[future]
            try:
                local_path = future.result()
                results.append(local_path)
            except Exception as e:
                logger.error(f"{repo_id} failed permanently: {e}")
                # Optionally append None or re-raise depending on your needs
                results.append(None)

    return results

class ProgressBar:
    """Custom progress bar implementation mimicking tqdm behavior."""

    _lock = threading.RLock()  # class-level lock

    @classmethod
    def get_lock(cls):
        return cls._lock

    @classmethod
    def set_lock(cls, lock):
        cls._lock = lock

    def __init__(self, iterable=None, total: Optional[int] = None, desc: str = "",
                 unit: str = "it", unit_scale: bool = False, **kwargs):
        self.iterable = iterable
        self.total = total or (len(iterable) if iterable is not None else None)
        self.unit = unit
        self.unit_scale = unit_scale
        self.desc = desc
        self.current = 0
        self.start_time = datetime.now()
        self._last_update = 0
        self._width = 50  # Width of the progress bar
        self._iterator = iter(iterable) if iterable is not None else None

    def __iter__(self):
        """Make ProgressBar iterable."""
        return self

    def __next__(self):
        """Iterate over the underlying iterable and update progress."""
        if self._iterator is None:
            raise StopIteration
        try:
            item = next(self._iterator)
            self.update(1)
            return item
        except StopIteration:
            self.close()
            raise

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
