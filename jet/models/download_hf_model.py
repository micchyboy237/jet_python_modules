from huggingface_hub import snapshot_download

repo_id = "mlx-community/Qwen3-1.7B-4bit-DWQ-053125"
cache_dir = "/Users/jethroestrada/.cache/huggingface/hub"
snapshot_download(repo_id=repo_id, cache_dir=cache_dir,
                  local_dir_use_symlinks=False, max_workers=4,
                  force_download=True)
print(f"Model repository downloaded to: {cache_dir}")
