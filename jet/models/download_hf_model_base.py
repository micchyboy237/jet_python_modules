import argparse
import os

from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser(
    description="Download a Hugging Face model repo by repo_id."
)
parser.add_argument(
    "repo_id",
    type=str,
    help="Repository ID (e.g., mlx-community/Ministral-3-3B-Instruct-2512-4bit)",
)
args = parser.parse_args()


def format_repo_cache_dir(repo_id: str) -> str:
    parts = repo_id.split("/")
    return f"models--{'--'.join(parts)}"


repo_cache_name = format_repo_cache_dir(args.repo_id)

models_base_dir = os.path.expanduser(f"~/.cache/pretrained_models/{repo_cache_name}")

snapshot_download(
    repo_id=args.repo_id,
    repo_type="model",
    local_dir=models_base_dir,
    local_dir_use_symlinks=False,
)
