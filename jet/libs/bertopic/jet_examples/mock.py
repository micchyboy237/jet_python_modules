import os
from typing import TypedDict, List
from sklearn.datasets import fetch_20newsgroups
from typing import Literal

from jet.utils.inspect_utils import get_entry_file_dir, get_entry_file_name
from jet.wordnet.text_chunker import chunk_texts_fast
from jet.file.utils import save_file
from jet.logger import logger

EMBED_MODEL = "embeddinggemma"

_sample_data_cache = None

class NewsGroupDocument(TypedDict):
    text: str
    target: int
    label: int
    category: str

class TargetInfo(TypedDict):
    label: int
    category: str

def fetch_newsgroups_samples(*, subset: Literal["train", "test", "all"] = "train", **kwargs) -> List[NewsGroupDocument]:
    """Load sample dataset from 20 newsgroups for topic modeling, with global cache."""
    global _sample_data_cache
    if _sample_data_cache is not None:
        logger.info(f"Reusing sample data cache ({len(_sample_data_cache)})")
        return _sample_data_cache

    logger.info("Loading 20 newsgroups dataset...")
    fetch_args = {
        "subset": subset,
        "remove": ("headers", "footers", "quotes"),
        "random_state": 42,
    }
    fetch_args.update(kwargs)
    newsgroups = fetch_20newsgroups(**fetch_args)

    # Iterate through the dataset to build the list of typed dictionaries
    samples = []
    for idx in range(len(newsgroups.data)):
        doc: NewsGroupDocument = {
            "text": newsgroups.data[idx],
            "target": newsgroups.target[idx],
            "label": newsgroups.target[idx],  # Add label field, same as target
            "category": newsgroups.target_names[newsgroups.target[idx]]
        }
        samples.append(doc)

    save_file(samples, f"{get_entry_file_dir()}/generated/{os.path.splitext(get_entry_file_name())[0]}/samples.json")

    return samples

def get_unique_categories(*, subset: Literal["train", "test", "all"] = "train", **kwargs) -> List[TargetInfo]:
    """
    Fetch unique categories and their corresponding labels from the 20 newsgroups dataset.
    
    Args:
        subset: The subset of the dataset to fetch ("train", "test", or "all"). Defaults to "train".
    
    Returns:
        A list of dictionaries containing unique label and category pairs.
    """
    newsgroups = fetch_20newsgroups(subset=subset, remove=("headers", "footers", "quotes"), **kwargs)
    unique_categories: List[TargetInfo] = [
        {"label": i, "category": newsgroups.target_names[i]}
        for i in sorted(set(newsgroups.target))
    ]

    save_file(unique_categories, f"{get_entry_file_dir()}/generated/{os.path.splitext(get_entry_file_name())[0]}/categories.json")

    return unique_categories

def load_sample_data(limit: int = 100, subset: Literal["train", "test", "all"] = "train") -> List[str]:
    """
    Fetch information from the 20newsgroups newsgroups and return as a list of typed dictionaries.
    
    Args:
        subset: The subset of the dataset to fetch ("train", "test", or "all"). Defaults to "train".
    
    Returns:
        A list of dictionaries containing the text, target label, and category name for each document.
    """
    # Fetch the dataset
    dataset = fetch_newsgroups_samples(subset=subset)
    
    # Prepare the result list
    documents: List[NewsGroupDocument] = [d["text"] for d in dataset]
    documents = documents[:limit]
    documents = chunk_texts_fast(
        documents,
        chunk_size=64,
        chunk_overlap=32,
        model=EMBED_MODEL,
    )
    _sample_data_cache = documents

    save_file(documents, f"{get_entry_file_dir()}/generated/{os.path.splitext(get_entry_file_name())[0]}/docs.json")

    return documents
