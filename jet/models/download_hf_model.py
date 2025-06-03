from huggingface_hub import snapshot_download

repo_id = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
cache_dir = "/Users/jethroestrada/.cache/huggingface/hub"
snapshot_download(repo_id=repo_id, cache_dir=cache_dir,
                  local_dir_use_symlinks=False, max_workers=4)
print(f"Model repository downloaded to: {cache_dir}")
