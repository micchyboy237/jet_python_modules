from pathlib import Path
import shutil
from typing import Union
from huggingface_hub import snapshot_download
import os
import glob

from jet.logger import logger
from jet.models.config import MODELS_CACHE_DIR, XET_CACHE_DIR
from jet.models.model_types import ModelType
from jet.models.download_onnx_model import download_onnx_model
from jet.models.utils import resolve_model_value


def remove_cache_locks(cache_dir: str = MODELS_CACHE_DIR) -> None:
    """
    Remove all lock files from the specified cache directory.

    Args:
        cache_dir (str): Path to the cache directory
    """
    try:
        lock_pattern = os.path.join(cache_dir, "**", "*.lock")
        lock_files = glob.glob(lock_pattern, recursive=True)
        for lock_file in lock_files:
            try:
                os.remove(lock_file)
                logger.debug(f"Removed lock file: {lock_file}")
            except OSError as e:
                logger.warning(
                    f"Failed to remove lock file {lock_file}: {str(e)}")
        if not lock_files:
            logger.debug("No lock files found in cache directory")
    except Exception as e:
        logger.error(f"Error while removing lock files: {str(e)}")
        raise


def remove_download_cache() -> None:
    shutil.rmtree(XET_CACHE_DIR, ignore_errors=True)

    remove_cache_locks()


def download_hf_model(repo_id: Union[str, ModelType], cache_dir: str = MODELS_CACHE_DIR):
    try:
        model_path = resolve_model_value(repo_id)
    except ValueError:
        model_path = repo_id

    remove_download_cache()

    snapshot_download(
        repo_id=model_path,
        cache_dir=cache_dir,
        # allow_patterns=[
        #     # "onnx/model.onnx",
        #     "onnx/model_qint8_arm64.onnx",
        #     # "onnx/model_quantized.onnx"
        # ],
        ignore_patterns=[
            "model.safetensors",
            "pytorch_model.bin",
            "tf_model.h5",
            "onnx/model.onnx",
            "onnx/model_bnb4.onnx",
            "onnx/model_fp16.onnx",
            "onnx/model_int8.onnx",
            "onnx/model_q4.onnx",
            "onnx/model_q4f16.onnx",
            "onnx/model_quantized.onnx",
            "onnx/model_uint8.onnx",
            "openvino/",
            "openvino/openvino_model.bin",
            "openvino/openvino_model.xml",
            "openvino/openvino_model_qint8_quantized.xml",
        ],
        local_dir_use_symlinks=False,
        force_download=True
    )

    download_onnx_model(repo_id)

    remove_download_cache()


if __name__ == "__main__":
    repo_id = "sentence-transformers/all-MiniLM-L6-v2"
    cache_dir = MODELS_CACHE_DIR

    logger.info(f"Downloading files from repo id: {repo_id}...")

    try:
        logger.info(f"Removing lock files from cache directory: {cache_dir}")
        download_hf_model(repo_id)

        logger.info("Download completed")
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        raise
