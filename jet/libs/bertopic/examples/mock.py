from typing import Optional, TypedDict, List
from sklearn.datasets import fetch_20newsgroups
from typing import Literal

from jet.code.markdown_types.markdown_parsed_types import HeaderDoc
from jet.code.markdown_utils._preprocessors import clean_markdown_links
from jet.utils.url_utils import clean_links
from jet.wordnet.text_chunker import chunk_texts, truncate_texts
from jet.file.utils import load_file
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

def load_sample_data(model: str = EMBED_MODEL, chunk_size: int = 128, chunk_overlap: int = 32, truncate: bool = False) -> List[str]:
    """Load sample dataset from local for topic modeling."""
    headers_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/search/playwright/generated/run_playwright_extract/top_isekai_anime_2025/all_headers.json"
    
    logger.info("Loading sample dataset...")
    headers_dict = load_file(headers_file)
    # headers: List[HeaderDoc] = [h for h_list in headers_dict.values() for h in h_list]
    headers: List[HeaderDoc] = headers_dict["https://gamerant.com/new-isekai-anime-2025"]
    documents = [f"{doc["header"]}\n\n{doc['content']}" for doc in headers]

    # Clean all links
    documents = [clean_markdown_links(doc) for doc in documents]
    documents = [clean_links(doc) for doc in documents]

    if not truncate:
        documents = chunk_texts(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            model=model,
        )
    else:
        documents = truncate_texts(
            documents,
            max_tokens=chunk_size,
            model=model,
            strict_sentences=True
        )

    return [doc for doc in documents if doc.strip()]

def load_news_sample_data(model: str = EMBED_MODEL, chunk_size: int = 128, chunk_overlap: int = 32) -> List[str]:
    """
    Fetch information from the 20newsgroups newsgroups and return as a list of typed dictionaries.
    
    Args:
        subset: The subset of the dataset to fetch ("train", "test", or "all"). Defaults to "train".
    
    Returns:
        A list of dictionaries containing the text, target label, and category name for each document.
    """
    # Load cache
    global _sample_data_cache
    if _sample_data_cache is not None:
        logger.info(f"Reusing sample data cache ({len(_sample_data_cache)})")
        return _sample_data_cache

    limit: int = 100
    subset: Literal["train", "test", "all"] = "train"

    # Fetch the dataset
    samples = fetch_newsgroups_samples(subset=subset)

    # Save the categories for reference
    # categories = get_unique_categories(samples)

    
    # Prepare the result list
    documents: List[str] = [d["text"] for d in samples]
    documents = documents[:limit]

    chunks = chunk_texts(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        model=model,
    )
    _sample_data_cache = chunks

    return chunks
