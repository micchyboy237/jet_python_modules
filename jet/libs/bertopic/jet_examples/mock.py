import os
from typing import Optional, TypedDict, List
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
    count: int

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

def get_unique_categories(samples: Optional[List[NewsGroupDocument]] = None, *, subset: Literal["train", "test", "all"] = "train", **kwargs) -> List[TargetInfo]:
    """
    Fetch unique categories, their corresponding labels, and their counts from the 20 newsgroups dataset or provided samples.
    
    Args:
        samples: Optional list of NewsGroupDocument to extract categories from. If None, fetches from dataset.
        subset: The subset of the dataset to fetch ("train", "test", or "all"). Defaults to "train".
    
    Returns:
        A list of dictionaries containing unique label, category, and count of occurrences.
    """
    if samples is not None:
        # Count occurrences of each (target, category) pair
        from collections import Counter
        category_counts = Counter((doc["target"], doc["category"]) for doc in samples)
        unique_pairs = sorted(category_counts.keys(), key=lambda x: x[0])
        unique_categories: List[TargetInfo] = [
            {"label": target, "category": category, "count": category_counts[(target, category)]}
            for target, category in unique_pairs
        ]
    else:
        newsgroups = fetch_20newsgroups(subset=subset, remove=("headers", "footers", "quotes"), **kwargs)
        # Count occurrences of each target
        from collections import Counter
        target_counts = Counter(newsgroups.target)
        unique_categories: List[TargetInfo] = [
            {"label": i, "category": newsgroups.target_names[i], "count": target_counts[i]}
            for i in sorted(set(newsgroups.target))
        ]
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

    # Save the categories for reference
    categories = get_unique_categories(dataset)

    save_file(categories, f"{get_entry_file_dir()}/generated/{os.path.splitext(get_entry_file_name())[0]}/categories.json")
    
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
