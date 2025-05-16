from huggingface_hub import hf_hub_download, HfApi
import os
import requests
from pathlib import Path

# Your EMBED_MODELS dictionary
EMBED_MODELS = {
    "nomic-embed-text": "nomic-ai/nomic-embed-text-v1.5",
    "mxbai-embed-large": "mixedbread-ai/mxbai-embed-large-v1",
    "granite-embedding": "ibm-granite/granite-embedding-30m-english",
    "granite-embedding:278m": "ibm-granite/granite-embedding-278m-multilingual",
    "all-minilm:22m": "sentence-transformers/all-MiniLM-L6-v2",
    "all-minilm:33m": "sentence-transformers/all-MiniLM-L12-v2",
    "snowflake-arctic-embed:33m": "Snowflake/snowflake-arctic-embed-s",
    "snowflake-arctic-embed:137m": "Snowflake/snowflake-arctic-embed-m-long",
    "snowflake-arctic-embed": "Snowflake/snowflake-arctic-embed-l",
    "paraphrase-multilingual": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "bge-large": "BAAI/bge-large-en-v1.5",
}

# Initialize Hugging Face API
hf_api = HfApi()


def download_readme(model_id, model_name, output_dir, overwrite=False):
    """Download README for a given model, try API first, then web scraping."""
    readme_path = output_dir / f"{model_name}_README.md"

    # Check if README already exists and overwrite is False
    if readme_path.exists() and not overwrite:
        print(f"README for {model_name} already exists, skipping...")
        return True

    # Try downloading README via Hugging Face API
    try:
        hf_hub_download(
            repo_id=model_id,
            filename="README.md",
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        # Rename the downloaded file to include model_name
        downloaded_file = output_dir / "README.md"
        if downloaded_file.exists():
            downloaded_file.rename(readme_path)
        print(f"Downloaded README for {model_name} via API")
        return True
    except Exception as e:
        print(f"API failed for {model_name}: {e}")

    # Fallback to web scraping
    try:
        url = f"https://huggingface.co/{model_id}/raw/main/README.md"
        response = requests.get(url)
        if response.status_code == 200:
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            print(f"Downloaded README for {model_name} via web")
            return True
        else:
            print(
                f"Web scraping failed for {model_name}: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"Web scraping failed for {model_name}: {e}")
        return False


def download_mlx_model_readmes(output_dir="hf_readmes", overwrite=False):
    # Convert output_dir to Path object and create directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Iterate over models and download READMEs
    for model_name, model_id in EMBED_MODELS.items():
        print(f"Processing {model_name} ({model_id})...")
        download_readme(model_id, model_name, output_dir, overwrite)

    print(f"All READMEs saved to {output_dir}")
