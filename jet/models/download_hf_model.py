from huggingface_hub import snapshot_download
import os
import glob

from jet.logger import logger


repo_id = "mixedbread-ai/mxbai-embed-large-v1"
cache_dir = "/Users/jethroestrada/.cache/huggingface/hub"


def remove_cache_locks(cache_dir: str) -> None:
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


logger.info(f"Downloading files from repo id: {repo_id}...")
try:
    logger.info(f"Removing lock files from cache directory: {cache_dir}")
    remove_cache_locks(cache_dir)

    snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        # allow_patterns=[
        #     # "onnx/model.onnx",
        #     "onnx/model_qint8_arm64.onnx",
        #     # "onnx/model_quantized.onnx"
        # ],
        ignore_patterns=[
            "model.safetensors",
            "pytorch_model.bin",
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
    logger.info("Download completed")
except Exception as e:
    logger.error(f"Download failed: {str(e)}")
    raise
