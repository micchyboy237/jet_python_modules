from huggingface_hub import snapshot_download

repo_id = "Qwen/Qwen3-Embedding-0.6B"
cache_dir = "/Users/jethroestrada/.cache/huggingface/hub"
snapshot_download(repo_id=repo_id, cache_dir=cache_dir,
                  local_dir_use_symlinks=False, max_workers=4)
print(f"Model repository downloaded to: {cache_dir}")
