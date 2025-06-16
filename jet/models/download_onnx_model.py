from huggingface_hub import snapshot_download
import logging
import subprocess
from pathlib import Path
from typing import List, Optional

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


def download_and_process_model(
    repo_id: str,
    cache_dir: str,
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
    # Replace with dynamic retrieval if needed
    snapshot_hash = "ce0834f22110de6d9222af7a7a03628121708969"
    snapshot_path = snapshot_dir / snapshot_hash
    source_path = snapshot_path / model_file
    target_path = snapshot_path / target_file

    # Download model
    logger.info(f"Downloading files from repo id: {repo_id}...")
    try:
        snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            allow_patterns=[model_file],
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
        logger.error(f"Download failed: {str(e)}")
        raise

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

        # ls -l <onnx_dir>
        run_shell_command(
            ["ls", "-l", str(snapshot_path / "onnx")],
            f"Listing directory {snapshot_path / 'onnx'}",
        )

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
    repo_id = "cross-encoder/ms-marco-MiniLM-L6-v2"
    cache_dir = "/Users/jethroestrada/.cache/huggingface/hub"
    download_and_process_model(repo_id, cache_dir)
