from huggingface_hub import HfApi, GitCommitInfo, snapshot_download
import logging
import subprocess
from pathlib import Path
from typing import List, Optional

from jet.models.config import MODELS_CACHE_DIR

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_shell_command(command: List[str], description: str) -> Optional[str]:
    """Execute a shell command and log the result."""
    try:
        logger.debug(f"Executing {description}: {' '.join(command)}")
        result = subprocess.run(
            command, check=True, capture_output=True, text=True
        )
        logger.debug(f"{description} output: {result.stdout}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"{description} failed: {e.stderr}")
        raise


def download_onnx_model(
    repo_id: str,
    cache_dir: str = MODELS_CACHE_DIR,
    model_file: str = "onnx/model_qint8_arm64.onnx",
    target_file: str = "onnx/model.onnx",
) -> None:
    """
    Download model files and execute post-download shell commands.

    Args:
        repo_id: Repository ID for Hugging Face model.
        cache_dir: Directory to cache downloaded files.
        model_file: Source model file path relative to snapshot directory.
        target_file: Target model file path relative to snapshot directory.
    """
    snapshot_dir = Path(cache_dir) / \
        f"models--{repo_id.replace('/', '--')}" / "snapshots"
    # Dynamically retrieve the latest snapshot hash for the repo
    api = HfApi()
    snapshots: List[GitCommitInfo] = api.list_repo_commits(repo_id)
    if not snapshots:
        raise RuntimeError(f"No snapshots found for repo {repo_id}")
    snapshot_hash = snapshots[0].commit_id
    snapshot_path = snapshot_dir / snapshot_hash

    # Check for any other folders under snapshot_dir and remove if not the current snapshot_hash
    if snapshot_dir.exists() and snapshot_dir.is_dir():
        for folder in snapshot_dir.iterdir():
            if folder.is_dir() and folder.name != snapshot_hash:
                logger.info(f"Removing old snapshot directory: {folder}")
                try:
                    import shutil
                    shutil.rmtree(folder)
                except Exception as e:
                    logger.error(f"Failed to remove {folder}: {e}")

    # Verify available files in the repository and select model file by priority
    logger.info(
        f"Checking available files in repo {repo_id} for snapshot {snapshot_hash}")
    try:
        repo_files = api.list_repo_files(
            repo_id=repo_id, revision=snapshot_hash)
        logger.debug(f"Available files: {repo_files}")

        # Define priority order for model files
        model_file_candidates = [
            "onnx/model_qint8_arm64.onnx",
            "onnx/model_quantized.onnx",
            "onnx/model.onnx",
            "model.onnx"
        ]
        selected_model_file = None
        for candidate in model_file_candidates:
            if candidate in repo_files:
                selected_model_file = candidate
                break

        if selected_model_file is None:
            logger.error(
                f"No suitable model file found in repository {repo_id} for snapshot {snapshot_hash}")
            raise FileNotFoundError(
                f"No suitable model file found in repository {repo_id} snapshot {snapshot_hash}.\nExpected one of {model_file_candidates}, but available files: {repo_files}"
            )
        logger.info(f"Selected model file: {selected_model_file}")
    except Exception as e:
        logger.error(f"Failed to list repository files: {str(e)}")
        raise

    source_path = snapshot_path / selected_model_file
    target_path = snapshot_path / target_file

    # Download only the selected model file
    logger.info(
        "Downloading file %s from repo id: %s...",
        source_path,
        selected_model_file
    )

    try:
        snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            allow_patterns=[selected_model_file],
            ignore_patterns=[
                "onnx/model_O1.onnx",
                "onnx/model_O2.onnx",
                "onnx/model_O3.onnx",
                "onnx/model_O4.onnx",
                "onnx/model_qint8_avx512.onnx",
                "onnx/model_qint8_avx512_vnni.onnx",
                "onnx/model_quint8_avx2.onnx",
            ],
            local_dir_use_symlinks=False,
            force_download=True,
        )
        logger.info("Download completed")
    except Exception as e:
        logger.error(f"Download failed for {selected_model_file}: {str(e)}")
        raise

    # Verify that the source model file exists
    if not source_path.exists():
        logger.error(
            f"Model file {source_path} does not exist after download.")
        raise FileNotFoundError(
            f"Model file {source_path} not found in snapshot {snapshot_hash}")

    # Execute shell commands
    try:
        # ls -l <source_path>
        run_shell_command(
            ["ls", "-l", str(source_path)],
            f"Listing file {source_path}",
        )

        # Check if target_path exists and log
        if target_path.exists():
            logger.warning(
                f"Target file {target_path} already exists. Overwriting with cp -f.")

        # Ensure target directory exists before copying
        target_dir = target_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)

        # cp -f -L <source_path> <target_path>
        run_shell_command(
            ["cp", "-f", "-L", str(source_path), str(target_path)],
            f"Copying {source_path} to {target_path}",
        )

        # rm <source_path>
        run_shell_command(
            ["rm", str(source_path)],
            f"Removing {source_path}",
        )

        # Remove all symlinks in onnx folder except model.onnx and their referenced files
        onnx_dir = snapshot_path / "onnx"
        if onnx_dir.exists() and onnx_dir.is_dir():
            for item in onnx_dir.iterdir():
                if item.name != "model.onnx" and item.is_symlink():
                    try:
                        # Get the real path of the symlink
                        real_path = item.resolve()
                        logger.info(
                            f"Removing symlink {item} pointing to {real_path}")
                        item.unlink()  # Remove the symlink
                        if real_path.exists():
                            logger.info(
                                f"Removing referenced file {real_path}")
                            real_path.unlink()  # Remove the referenced file
                    except Exception as e:
                        logger.error(
                            f"Failed to remove symlink {item} or its target: {e}")

        # ls -l <onnx_dir> only if onnx_dir exists
        if onnx_dir.exists() and onnx_dir.is_dir():
            run_shell_command(
                ["ls", "-l", str(onnx_dir)],
                f"Listing directory {onnx_dir}",
            )
        else:
            logger.info(
                f"onnx/ directory does not exist, skipping ls -l command")

        # du -sh <target_path>
        run_shell_command(
            ["du", "-sh", str(target_path)],
            f"Checking size of {target_path}",
        )

        logger.info("All shell commands executed successfully")
    except Exception as e:
        logger.error(f"Shell command execution failed: {str(e)}")
        raise


if __name__ == "__main__":
    repo_id = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    cache_dir = MODELS_CACHE_DIR
    download_onnx_model(repo_id, cache_dir)
