from huggingface_hub import snapshot_download
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

repo_id = "sentence-transformers/static-retrieval-mrl-en-v1"
cache_dir = "/Users/jethroestrada/.cache/huggingface/hub"

logger.info(f"Downloading files from repo id: {repo_id}...")
try:
    snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        allow_patterns=["tokenizer.json"],
        local_dir_use_symlinks=False,
        force_download=True
    )
    logger.info("Download completed")
except Exception as e:
    logger.error(f"Download failed: {str(e)}")
    raise
