from typing import Literal, TypedDict

from jet.code.html_utils import convert_dl_blocks_to_md, preprocess_html

# from jet.code.extraction.sentence_extraction import extract_sentences
from jet.code.markdown_types.markdown_parsed_types import MarkdownToken
from jet.code.markdown_utils import base_parse_markdown, derive_by_header_hierarchy
from jet.code.markdown_utils._converters import (
    convert_html_to_markdown,
    convert_markdown_to_text,
)
from jet.file.utils import load_file
from jet.logger import logger
from jet.wordnet.text_chunker import (
    ChunkResult,
    chunk_texts,
    chunk_texts_with_data,
    truncate_texts,
)
from sklearn.datasets import fetch_20newsgroups

EMBED_MODEL = "embeddinggemma"

_sample_data_cache = None


class ChunkResultMeta(TypedDict):
    """Metadata for document chunks.

    Attributes:
        doc_id: Document ID (same for all chunks of a document).
        doc_index: Document index in the source dataset.
        header: Header text (e.g., '### Title').
        level: Header level (e.g., 2 → '##').
        parent_header: Parent section header (e.g., '## Parent').
        parent_level: Parent header level (e.g., 2 → '##').
        source: File path, URL, or other source reference.
        tokens: List of parsed markdown tokens for this chunk.
    """

    doc_id: str
    doc_index: int
    header: str
    level: int | None
    parent_header: str | None
    parent_level: int | None
    source: str | None
    tokens: list[MarkdownToken]


class ChunkResultWithMeta(ChunkResult):
    """Chunk data extended with metadata.

    Attributes:
        meta: Metadata containing headers, structure, and source info.
    """

    meta: ChunkResultMeta


class NewsGroupDocument(TypedDict):
    text: str
    target: int
    label: int
    category: str


class TargetInfo(TypedDict):
    label: int
    category: str
    count: int


def fetch_newsgroups_samples(
    *, subset: Literal["train", "test", "all"] = "train", **kwargs
) -> list[NewsGroupDocument]:
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
            "category": newsgroups.target_names[newsgroups.target[idx]],
        }
        samples.append(doc)

    return samples


def get_unique_categories(
    samples: list[NewsGroupDocument] | None = None,
    *,
    subset: Literal["train", "test", "all"] = "train",
    **kwargs,
) -> list[TargetInfo]:
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
        unique_categories: list[TargetInfo] = [
            {
                "label": target,
                "category": category,
                "count": category_counts[(target, category)],
            }
            for target, category in unique_pairs
        ]
    else:
        newsgroups = fetch_20newsgroups(
            subset=subset, remove=("headers", "footers", "quotes"), **kwargs
        )
        # Count occurrences of each target
        from collections import Counter

        target_counts = Counter(newsgroups.target)
        unique_categories: list[TargetInfo] = [
            {
                "label": i,
                "category": newsgroups.target_names[i],
                "count": target_counts[i],
            }
            for i in sorted(set(newsgroups.target))
        ]
    return unique_categories


def load_sample_md_doc() -> str:
    html = load_file(
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/search/playwright/generated/run_playwright_extract/top_isekai_anime_2025/https_gamerant_com_new_isekai_anime_2025/page.html"
    )
    html = convert_dl_blocks_to_md(html)
    html = preprocess_html(html, excludes=["nav", "footer"])
    md_content = convert_html_to_markdown(html, ignore_links=True)
    return md_content


def load_sample_data(
    model: str = EMBED_MODEL,
    chunk_size: int = 128,
    chunk_overlap: int = 0,
    truncate: bool = False,
    convert_plain_text: bool = False,
    includes: list[str] = [],
) -> list[str]:
    """Load sample dataset from local for topic modeling."""
    html = load_file(
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/search/playwright/generated/run_playwright_extract/top_isekai_anime_2025/https_gamerant_com_new_isekai_anime_2025/page.html"
    )
    html = convert_dl_blocks_to_md(html)
    html = preprocess_html(html, excludes=["nav", "footer"])

    md_content = convert_html_to_markdown(html, ignore_links=True)
    headings = derive_by_header_hierarchy(md_content, ignore_links=True)
    header_contents = [
        f"{header['header']}\n{header['content']}"
        for header in headings
        if header["content"]
    ]
    texts = header_contents

    # md_content = convert_html_to_markdown(html, ignore_links=True)
    # md_tokens = base_parse_markdown(md_content)
    # texts = [token['content'] for token in md_tokens if token['content']]

    # headings: List[HtmlHeaderDoc] = extract_header_hierarchy(html, includes=includes)
    # header_contents = [f"{header["header"]}\n{header["content"]}" for header in headings if header['content']]
    # texts = header_contents

    if convert_plain_text:
        # Preprocess markdown to plain text
        texts = [convert_markdown_to_text(md_content) for md_content in texts]
    # else:
    #     texts = header_md_contents

    if not truncate:
        # documents = chunk_texts(
        #     texts,
        #     chunk_size=chunk_size,
        #     chunk_overlap=chunk_overlap,
        #     model=model,
        #     strict_sentences=True,
        # )
        documents = texts
    else:
        documents = truncate_texts(
            texts, max_tokens=chunk_size, model=model, strict_sentences=True
        )

    return [doc for doc in documents if doc.strip()]


def load_sample_data_with_info(
    model: str = EMBED_MODEL,
    chunk_size: int = 128,
    chunk_overlap: int = 0,
    truncate: bool = False,
    convert_plain_text: bool = False,
    includes: list[str] = [],
) -> list[ChunkResultWithMeta]:
    """
    Load sample dataset from local for topic modeling, returning chunk results with section meta information.
    If truncate is True, only keep chunks where chunk_index == 0.
    """
    html = load_file(
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/search/playwright/generated/run_playwright_extract/top_isekai_anime_2025/https_gamerant_com_new_isekai_anime_2025/page.html"
    )
    html = convert_dl_blocks_to_md(html)
    html = preprocess_html(html, excludes=["nav", "footer"])

    md_content = convert_html_to_markdown(html, ignore_links=True)
    headings = derive_by_header_hierarchy(md_content, ignore_links=True)
    header_contents = [
        f"{header['header']}\n{header['content']}"
        for header in headings
        if header["content"]
    ]
    texts = header_contents

    # md_content = convert_html_to_markdown(html, ignore_links=True)
    md_tokens = base_parse_markdown(md_content)
    # texts = [token['content'] for token in md_tokens if token['content']]

    # headings: List[HtmlHeaderDoc] = extract_header_hierarchy(html, includes=includes)
    # header_contents = [f"{header["header"]}\n{header["content"]}" for header in headings if header['content']]
    # texts = header_contents

    if convert_plain_text:
        # Preprocess markdown to plain text
        texts = [convert_markdown_to_text(md_content) for md_content in texts]
    # else:
    #     texts = header_sentences
    doc_ids = [header["id"] for header in headings]

    # Map header id to header metadata (assuming doc_ids are unique and order-aligned)
    header_id_to_meta: dict[str, ChunkResultMeta] = {}
    for header in headings:
        header_meta: ChunkResultMeta = {
            "doc_id": header["id"],
            "doc_index": header["doc_index"],
            "header": header["header"],
            "level": header.get("level"),
            "parent_header": header.get("parent_header"),
            "parent_level": header.get("parent_level"),
            "source": None,
            "tokens": md_tokens,
        }
        header_id_to_meta[header["id"]] = header_meta

    base_chunks: list[ChunkResult] = chunk_texts_with_data(
        texts,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        model=model,
        ids=doc_ids,
        strict_sentences=True,
    )

    # Attach section meta-data from the corresponding HeaderDoc to each chunk
    enriched_chunks: list[ChunkResultWithMeta] = []
    for chunk in base_chunks:
        doc_id = chunk["doc_id"]
        if doc_id in header_id_to_meta:
            chunk_meta = header_id_to_meta[doc_id]
        else:
            # Safely fill with correct types for fields
            chunk_meta: ChunkResultMeta = {
                "doc_id": doc_id,
                "doc_index": -1,
                "header": "",
                "level": None,
                "parent_header": None,
                "parent_level": None,
                "source": None,
                "tokens": [],
            }
        chunk_with_meta: ChunkResultWithMeta = {**chunk, "meta": chunk_meta}
        enriched_chunks.append(chunk_with_meta)

    if truncate:
        enriched_chunks = [
            chunk for chunk in enriched_chunks if chunk.get("chunk_index", 0) == 0
        ]

    return enriched_chunks


def load_sample_jobs(
    model: str = EMBED_MODEL,
    chunk_size: int = 128,
    chunk_overlap: int = 0,
    truncate: bool = False,
    convert_plain_text: bool = False,
    includes: list[str] = [],
) -> list[str]:
    """Load sample jobs from local for topic modeling."""
    from shared.data_types.job import JobData

    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Apps/my-jobs/saved/jobs.json"
    data: list[JobData] = load_file(data_file)

    sentences = [
        "\n".join(
            [
                f"## {item['title']}\n",
                item["details"],
                "\n\nTechnology Stack:\n",
                "\n".join(
                    [
                        f"- {tech}"
                        for tech in sorted(
                            item["entities"]["technology_stack"], key=str.lower
                        )
                    ]
                ),
                "\n\nTags:\n",
                "\n".join(
                    [f"- {tech}" for tech in sorted(item["tags"], key=str.lower)]
                ),
            ]
        )
        for item in data
    ]
    logger.info(f"Number of sentences: {len(sentences)}")

    # texts = [token['content'] for sentence in sentences for token in base_parse_markdown(sentence)]
    texts = sentences

    return texts


def load_sample_text() -> str:
    """Load sample dataset from local for topic modeling."""
    html = load_file(
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/search/playwright/generated/run_playwright_extract/top_isekai_anime_2025/https_gamerant_com_new_isekai_anime_2025/page.html"
    )

    md_content = convert_html_to_markdown(html, ignore_links=True)

    return md_content


def load_news_sample_data(
    model: str = EMBED_MODEL, chunk_size: int = 128, chunk_overlap: int = 32
) -> list[str]:
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
    documents: list[str] = [d["text"] for d in samples]
    documents = documents[:limit]

    chunks = chunk_texts(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        model=model,
    )
    _sample_data_cache = chunks

    return chunks
