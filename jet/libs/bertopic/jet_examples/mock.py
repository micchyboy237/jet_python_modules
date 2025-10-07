import os
from jet.utils.inspect_utils import get_entry_file_dir, get_entry_file_name
from jet.wordnet.text_chunker import chunk_texts_fast
from jet.file.utils import save_file
from jet.logger import logger

EMBED_MODEL = "embeddinggemma"

_sample_data_cache = None

def load_sample_data(limit: int = 100, **kwargs):
    """Load sample dataset from 20 newsgroups for topic modeling, with global cache."""
    global _sample_data_cache
    if _sample_data_cache is not None:
        return _sample_data_cache
    from sklearn.datasets import fetch_20newsgroups
    logger.info("Loading 20 newsgroups dataset...")
    fetch_args = {
        "subset": "all",
        "remove": ("headers", "footers", "quotes"),
        "random_state": 42,
    }
    fetch_args.update(kwargs)
    newsgroups = fetch_20newsgroups(**fetch_args)
    documents = newsgroups.data[:limit]
    documents = chunk_texts_fast(
        documents,
        chunk_size=128,
        chunk_overlap=32,
        model=EMBED_MODEL,
    )
    _sample_data_cache = documents
    
    save_file(documents, f"{get_entry_file_dir()}/generated/{os.path.splitext(get_entry_file_name())[0]}/docs.json")

    return documents
