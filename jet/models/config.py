import os
from typing import Optional

HUGGINGFACE_BASE_DIR = os.path.expanduser("~/.cache/huggingface")


def get_hf_token(cache_dir: str = HUGGINGFACE_BASE_DIR) -> Optional[str]:
    """Retrieve the Hugging Face token from the token file or environment variable.

    Returns:
        Optional[str]: The Hugging Face token if found, else None.
    """
    token_path = os.path.join(cache_dir, "token")
    try:
        if os.path.exists(token_path):
            with open(token_path, "r") as f:
                return f.read().strip()
        return os.getenv("HF_TOKEN")
    except Exception:
        return None


MODELS_CACHE_DIR = os.path.join(HUGGINGFACE_BASE_DIR, "hub")
XET_CACHE_DIR = os.path.join(HUGGINGFACE_BASE_DIR, "xet")
HF_TOKEN: Optional[str] = get_hf_token()
