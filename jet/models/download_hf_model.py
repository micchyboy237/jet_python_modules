from huggingface_hub import snapshot_download
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

repo_id = "cross-encoder/ms-marco-MiniLM-L6-v2"
cache_dir = "/Users/jethroestrada/.cache/huggingface/hub"

logger.info(f"Downloading files from repo id: {repo_id}...")
try:
    snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        allow_patterns=[
            # "onnx/model.onnx",
            "onnx/model_qint8_arm64.onnx",
            # "onnx/model_quantized.onnx"
        ],
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
        force_download=True
    )
    logger.info("Download completed")
except Exception as e:
    logger.error(f"Download failed: {str(e)}")
    raise
