import asyncio
import re
import string
import numpy as np

from collections import defaultdict
from typing import DefaultDict, List, Set, TypedDict, Dict
from jet.code.html_utils import preprocess_html
from jet.code.markdown_types.markdown_parsed_types import HeaderDoc, HeaderSearchResult
from jet.code.markdown_utils._converters import convert_html_to_markdown
from jet.code.markdown_utils._markdown_analyzer import analyze_markdown
from jet.code.markdown_utils._markdown_parser import base_parse_markdown, derive_by_header_hierarchy, parse_markdown
from jet.code.markdown_utils._preprocessors import clean_markdown_links, link_to_text_ratio
from jet.logger import logger
from jet.logger.config import colorize_log
from jet.models.embeddings.base import generate_embeddings
# from jet.models.model_registry.transformers.mlx_model_registry import MLXModelRegistry
from jet.llm.mlx.remote import generation as gen
from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry
from jet.models.model_types import EmbedModelType, LLMModelType
from jet.models.utils import resolve_model_value
from jet.models.tokenizer.base import get_tokenizer_fn, count_tokens
from jet.scrapers.hrequests_utils import scrape_urls
from jet.scrapers.utils import scrape_links, search_data
from jet.vectors.semantic_search.header_vector_search import search_headers
from jet.wordnet.analyzers.text_analysis import calculate_mtld, calculate_mtld_category


PROMPT_TEMPLATE = """\
Context information is below.
---------------------
{context}
---------------------

Given the context information, answer the query.

Query: {query}
"""

HIGH_QUALITY_SCORE = 0.6
MEDIUM_QUALITY_SCORE = 0.4
TARGET_HIGH_SCORE_TOKENS = 4000
TARGET_MEDIUM_SCORE_TOKENS = 10000


def format_sub_dir(text: str) -> str:
    return text.lower().strip('.,!?').replace(' ', '_').replace('.', '_').replace(',', '_').replace('!', '_').replace('?', '_').strip()


def format_sub_source_dir(source: str) -> str:
    """Format a source (URL or file path) into a directory name."""
    clean_source = re.sub(r'^(https?://|www\.)|(\?.*)', '', source)
    clean_source = clean_source.replace(os.sep, '_')
    trans_table = str.maketrans({p: '_' for p in string.punctuation})
    formatted = clean_source.translate(trans_table).lower()
    formatted = re.sub(r'_+', '_', formatted)
    return formatted.strip('_')


def sort_urls_by_high_and_medium_score_tokens(results: List[HeaderSearchResult]) -> List[str]:
    # Group results by URL and calculate total medium_score_tokens per URL
    url_medium_score_tokens = defaultdict(int)
    for result in results:
        url = result["metadata"].get("source", "Unknown")
        if result["score"] >= MEDIUM_QUALITY_SCORE and result.get("mtld_category") != "very_low":
            url_medium_score_tokens[url] += result["metadata"].get(
                "num_tokens", 0)

    # Calculate high and medium score tokens per URL
    url_score_tokens = defaultdict(
        lambda: {"high_score_tokens": 0, "medium_score_tokens": 0}
    )

    sorted_urls = sorted(
        url_score_tokens.keys(),
        key=lambda url: (url_score_tokens[url]["high_score_tokens"],
                         url_score_tokens[url]["medium_score_tokens"]),
        reverse=True
    )

    return sorted_urls


def sort_search_results_by_url_and_category(results: List[HeaderSearchResult], sorted_urls: List[str]):
    """
    Sorts results in three stages:
    1. Results with score >= HIGH_QUALITY_SCORE, sorted by url order in sorted_urls, then by score descending within each url.
    2. Results with score >= MEDIUM_QUALITY_SCORE but < HIGH_QUALITY_SCORE, sorted by score descending.
    3. Results with score < MEDIUM_QUALITY_SCORE, sorted by score descending.
    Returns the concatenated list.
    """
    # Stage 1: Get results with score >= HIGH_QUALITY_SCORE, grouped by url order
    url_to_results = {url: [] for url in sorted_urls}
    high_score_results = []
    medium_score_results = []
    low_score_results = []
    for r in results:
        url = r["metadata"]["source"]
        if r["score"] >= HIGH_QUALITY_SCORE and url in url_to_results:
            url_to_results[url].append(r)
        elif r["score"] >= HIGH_QUALITY_SCORE:
            # If url not in sorted_urls, treat as extra at end
            high_score_results.append(r)
        elif r["score"] >= MEDIUM_QUALITY_SCORE:
            medium_score_results.append(r)
        else:
            low_score_results.append(r)

    # Collect high score results in url order, sorting by score within each url
    sorted_high_score = []
    for url in sorted_urls:
        url_group = url_to_results[url]
        url_group_sorted = sorted(
            url_group, key=lambda r: r["score"], reverse=True)
        sorted_high_score.extend(url_group_sorted)
    # Add any >= HIGH_QUALITY_SCORE results whose url wasn't in sorted_urls, sorted by score
    if high_score_results:
        sorted_high_score.extend(
            sorted(high_score_results, key=lambda r: r["score"], reverse=True))

    # Stage 2: Medium score results (MEDIUM_QUALITY_SCORE <= score < HIGH_QUALITY_SCORE), sort by score descending
    sorted_medium_score = sorted(
        medium_score_results, key=lambda r: r["score"], reverse=True)

    # Stage 3: Low score results (score < MEDIUM_QUALITY_SCORE), sort by score descending
    sorted_low_score = sorted(
        low_score_results, key=lambda r: r["score"], reverse=True)

    return sorted_high_score + sorted_medium_score + sorted_low_score


def group_results_by_source_for_llm_context(
    results: List[HeaderSearchResult]
) -> str:
    def strip_hashtags(text: str) -> str:
        if text:
            return text.lstrip('#').strip()
        return text

    # Initialize tokenizer
    tokenizer = get_tokenizer_fn(
        "llama-3.2-3b-instruct-4bit", add_special_tokens=False, remove_pad_tokens=True)
    separator = "\n\n"
    separator_tokens = len(tokenizer.encode(separator))

    # Calculate high and medium score tokens per URL
    url_score_tokens = defaultdict(
        lambda: {"high_score_tokens": 0, "medium_score_tokens": 0}
    )
    for result in results:
        url = result["metadata"].get("source", "Unknown")
        if result["score"] >= HIGH_QUALITY_SCORE:
            url_score_tokens[url]["high_score_tokens"] += result["metadata"].get(
                "num_tokens", 0)
        elif result["score"] >= MEDIUM_QUALITY_SCORE:
            url_score_tokens[url]["medium_score_tokens"] += result["metadata"].get(
                "num_tokens", 0)

    # Sort URLs by high_score_tokens, then medium_score_tokens (descending)
    sorted_urls = sorted(
        url_score_tokens.keys(),
        key=lambda url: (url_score_tokens[url]["high_score_tokens"],
                         url_score_tokens[url]["medium_score_tokens"]),
        reverse=True
    )

    # Group results by URL and sort within each URL by score
    grouped_temp: DefaultDict[str,
                              List[HeaderSearchResult]] = defaultdict(list)
    seen_header_text: DefaultDict[str, Set[str]] = defaultdict(set)
    for result in results:
        url = result["metadata"].get("source", "Unknown")
        grouped_temp[url].append(result)

    context_blocks = []
    for url in sorted_urls:
        docs = sorted(grouped_temp[url],
                      key=lambda x: x["score"], reverse=True)
        block = f"<!-- Source: {url} -->\n\n"
        seen_header_text_in_block = set()

        # Group by doc_index and header to handle overlaps
        grouped_by_header: DefaultDict[tuple[int, str],
                                       List[HeaderSearchResult]] = defaultdict(list)
        for doc in sorted(docs, key=lambda x: (x["metadata"].get("doc_index", 0), x["metadata"].get("start_idx", 0))):
            doc_index = doc["metadata"].get("doc_index", 0)
            header = doc.get("header", "") or ""
            grouped_by_header[(doc_index, header)].append(doc)

        for (doc_index, header), chunks in grouped_by_header.items():
            parent_header = chunks[0].get("parent_header", "None")
            parent_level = chunks[0]["metadata"].get("parent_level", None)
            doc_level = chunks[0]["metadata"].get(
                "level", 0) if chunks[0]["metadata"].get("level") is not None else 0
            parent_header_key = strip_hashtags(
                parent_header) if parent_header and parent_header != "None" else None
            header_key = strip_hashtags(header) if header else None

            # Check for matching child headers to avoid redundant parent headers
            has_matching_child = any(
                strip_hashtags(d.get("header", "")) == parent_header_key
                for d in docs
                if d.get("header") and d["metadata"].get("level", 0) >= 0
            )

            # Check if parent_header appears as a header in any other chunk within the same source
            has_matching_child = any(
                strip_hashtags(d.get("header", "")) == parent_header_key
                for d in docs
                if d.get("header") and strip_hashtags(d.get("header", "")) != header_key
            )

            # Add parent header only if it appears as a header in another chunk and hasn't been added
            if parent_header_key and parent_level is not None and has_matching_child and parent_header_key not in seen_header_text_in_block:
                block += f"{parent_header}\n\n"
                seen_header_text_in_block.add(parent_header_key)

            # Add header if it hasn't been added
            if header_key and header_key not in seen_header_text_in_block and doc_level >= 0:
                block += f"{header}\n\n"
                seen_header_text_in_block.add(header_key)
                seen_header_text[url].add(header_key)

            # Sort chunks by start_idx and merge overlapping or adjacent chunks
            chunks.sort(key=lambda x: x["metadata"]["start_idx"])
            merged_content = ""
            start_idx = chunks[0]["metadata"]["start_idx"]
            end_idx = chunks[0]["metadata"]["end_idx"]
            current_content = chunks[0]["content"]
            merged_content = current_content

            for next_chunk in chunks[1:]:
                next_start = next_chunk["metadata"]["start_idx"]
                next_end = next_chunk["metadata"]["end_idx"]
                next_content = next_chunk["content"]
                if not isinstance(next_content, str):
                    logger.debug(
                        f"Non-string content in chunk for source: {url}, doc_index: {doc_index}, type: {type(next_content)}. Converting to string.")
                    next_content = str(next_content) if next_content else ""

                # Merge if chunks overlap or are adjacent
                if next_start <= end_idx + 1:
                    overlap = end_idx - next_start + 1 if next_start <= end_idx else 0
                    additional_content = next_content[overlap:
                                                      ] if overlap > 0 else next_content
                    merged_content += additional_content
                    end_idx = max(end_idx, next_end)
                else:
                    # Append merged content to block
                    block += merged_content + "\n\n"
                    # Start new merged chunk
                    merged_content = next_content
                    start_idx = next_start
                    end_idx = next_end

            # Append the last merged chunk
            block += merged_content + "\n\n"

        block_tokens = len(tokenizer.encode(block))
        if block_tokens > len(tokenizer.encode(f"<!-- Source: {url} -->\n\n")):
            context_blocks.append(block.strip())
        else:
            logger.warning(
                f"Empty block for {url} after processing; skipping.")

    result = "\n\n".join(context_blocks)
    final_token_count = len(tokenizer.encode(result))
    logger.debug(
        f"Grouped context created with {final_token_count} tokens for {len(grouped_temp)} sources")
    return result


# Helper function to create list of dicts for URLs
def create_url_dict_list(urls: List[str], search_results: List[HeaderSearchResult]) -> List[dict]:
    # Calculate stats per URL from search_results
    url_stats = defaultdict(
        lambda: {
            "high_score_tokens": 0,
            "high_score_headers": 0,
            "medium_score_tokens": 0,
            "medium_score_headers": 0,
            "headers": 0,
            "max_score": float('-inf'),
            "min_score": float('inf')
        })
    for result in search_results:
        url = result["metadata"].get("source", "Unknown")
        score = result["score"]
        url_stats[url]["headers"] += 1
        url_stats[url]["max_score"] = max(url_stats[url]["max_score"], score)
        url_stats[url]["min_score"] = min(url_stats[url]["min_score"], score)
        if result["score"] >= HIGH_QUALITY_SCORE:
            url_stats[url]["high_score_tokens"] += result["metadata"].get(
                "num_tokens", 0)
            url_stats[url]["high_score_headers"] += 1
        elif result["score"] >= MEDIUM_QUALITY_SCORE:
            url_stats[url]["medium_score_tokens"] += result["metadata"].get(
                "num_tokens", 0)
            url_stats[url]["medium_score_headers"] += 1

    return [
        {
            "url": url,
            "max_score": url_stats[url]["max_score"],
            "min_score": url_stats[url]["min_score"],
            "high_score_tokens": url_stats[url]["high_score_tokens"],
            "high_score_headers": url_stats[url]["high_score_headers"],
            "medium_score_tokens": url_stats[url]["medium_score_tokens"],
            "medium_score_headers": url_stats[url]["medium_score_headers"],
            "headers": url_stats[url]["headers"],
        }
        for url in urls
    ]


class WebDeepSearchResult(TypedDict):
    query: str
    search_engine_results: List[Dict]
    started_urls: List[str]
    searched_urls: List[str]
    high_score_urls: List[Dict]
    header_docs: List[HeaderDoc]
    search_results: List[HeaderSearchResult]
    sorted_search_results: List[HeaderSearchResult]
    filtered_results: List[HeaderSearchResult]
    filtered_urls: List[Dict]
    context: str
    response_text: str
    token_info: Dict[str, int]

async def web_deep_search(query: str) -> WebDeepSearchResult:
    """Main function to perform web deep search and return structured results."""
    embed_model: EmbedModelType = "all-MiniLM-L6-v2"
    llm_model: LLMModelType = "llama-3.2-3b-instruct-4bit"
    max_tokens = 2000
    use_cache = True
    urls_limit = 10

    top_k = None
    threshold = 0.0
    chunk_size = 200
    chunk_overlap = 50
    merge_chunks = False

    search_engine_results = search_data(query, use_cache=use_cache)
    urls = [r["url"] for r in search_engine_results][:urls_limit]

    html_list = []
    header_docs: List[HeaderDoc] = []
    search_results: List[HeaderSearchResult] = []

    headers_total_tokens = 0
    headers_high_score_tokens = 0
    headers_medium_score_tokens = 0
    headers_mtld_score_average = 0

    all_started_urls = []
    all_completed_urls = []
    all_searched_urls = []
    all_urls_with_high_scores = []
    all_urls_with_low_scores = []

    async for url, status, html in scrape_urls(urls, show_progress=True):
        if status == "started":
            all_started_urls.append(url)
        elif status == "completed" and html:
            all_completed_urls.append(url)
            html_list.append(html)

            links = set(scrape_links(html, url))
            links = [link for link in links if (
                link != url if isinstance(link, str) else link["url"] != url)]

            doc_markdown = convert_html_to_markdown(html, ignore_links=False)
            doc_analysis = analyze_markdown(doc_markdown)
            doc_markdown_tokens = base_parse_markdown(doc_markdown)

            original_docs: List[HeaderDoc] = derive_by_header_hierarchy(
                doc_markdown, ignore_links=True)

            for doc in original_docs:
                doc["source"] = url

            sub_results = list(
                search_headers(
                    original_docs,
                    query,
                    top_k=top_k,
                    threshold=threshold,
                    embed_model=embed_model,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    tokenizer_model=embed_model,
                    merge_chunks=merge_chunks
                )
            )
            all_searched_urls.append(url)

            # Add ltr_ratio using link_to_text_ratio on each result by content
            filtered_sub_results = []
            for result in sub_results:
                ltr = link_to_text_ratio(result["content"])
                result["metadata"]["ltr_ratio"] = ltr
                mtld_result = calculate_mtld(result["content"])
                result["metadata"]["mtld"] = mtld_result
                result["metadata"]["mtld_category"] = calculate_mtld_category(
                    mtld_result)
                if (
                    result["score"] >= MEDIUM_QUALITY_SCORE
                    and result["metadata"]["mtld_category"] != "very_low"
                ):
                    filtered_sub_results.append(result)

            sub_total_tokens = sum(
                result["metadata"]["num_tokens"] for result in filtered_sub_results)

            sub_high_score_tokens = sum(
                result["metadata"]["num_tokens"]
                for result in filtered_sub_results
                if result["score"] >= HIGH_QUALITY_SCORE
            )

            sub_medium_score_tokens = sum(
                result["metadata"]["num_tokens"]
                for result in filtered_sub_results
                if (
                    result["score"] >= MEDIUM_QUALITY_SCORE
                    and result["score"] < HIGH_QUALITY_SCORE
                )
            )

            sub_mtld_score_values = [
                calculate_mtld(result["content"])
                for result in filtered_sub_results
                if (
                    result["score"] >= HIGH_QUALITY_SCORE
                    and calculate_mtld_category(calculate_mtld(result["content"]))
                )
            ]
            sub_mtld_score_average = (
                sum(sub_mtld_score_values) /
                len(sub_mtld_score_values)
                if sub_mtld_score_values else 0
            )

            header_docs.extend(original_docs)
            search_results.extend(filtered_sub_results)
            if sub_high_score_tokens > 0:
                all_urls_with_high_scores.append(url)
            else:
                all_urls_with_low_scores.append(url)

            headers_total_tokens += sub_total_tokens
            headers_high_score_tokens += sub_high_score_tokens
            headers_medium_score_tokens += sub_medium_score_tokens
            headers_mtld_score_average += round(
                sub_mtld_score_average, 2)

            # Stop processing if either high-score tokens reach TARGET_HIGH_SCORE_TOKENS
            # or combined high and medium-score tokens reach TARGET_MEDIUM_SCORE_TOKENS
            if headers_high_score_tokens >= TARGET_HIGH_SCORE_TOKENS or \
               (headers_high_score_tokens + headers_medium_score_tokens) >= TARGET_MEDIUM_SCORE_TOKENS:
                logger.info(
                    f"Stopping processing: {headers_high_score_tokens} high-score tokens "
                    f"and {headers_medium_score_tokens} medium-score tokens collected from source: {url}.")
                break

    # Sort search_results by score then update rank
    search_results = sorted(
        search_results, key=lambda x: x["score"], reverse=True)
    for i, result in enumerate(search_results, 1):
        result["rank"] = i

    # Calculate high_score_tokens, medium_score_tokens, and header_count per URL
    url_stats = defaultdict(
        lambda: {"high_score_tokens": 0, "medium_score_tokens": 0, "header_count": 0})
    for result in search_results:
        url = result["metadata"].get("source", "Unknown")
        if result["score"] >= HIGH_QUALITY_SCORE:
            url_stats[url]["high_score_tokens"] += result["metadata"].get(
                "num_tokens", 0)
            url_stats[url]["header_count"] += 1
        elif result["score"] >= MEDIUM_QUALITY_SCORE:
            url_stats[url]["medium_score_tokens"] += result["metadata"].get(
                "num_tokens", 0)
            url_stats[url]["header_count"] += 1

    # Filter URLs with high_score_tokens > 0 or medium_score_tokens > 0 and format as list of dicts
    sorted_urls = [
        {
            "url": url,
            "high_score_tokens": stats["high_score_tokens"],
            "medium_score_tokens": stats["medium_score_tokens"],
            "header_count": stats["header_count"]
        }
        for url, stats in sorted(
            url_stats.items(),
            key=lambda x: (x[1]["high_score_tokens"],
                           x[1]["medium_score_tokens"]),
            reverse=True
        )
        if stats["high_score_tokens"] > 0 or stats["medium_score_tokens"] > 0
    ]

    # Sort URLs by high_score_tokens, then medium_score_tokens (descending)
    sorted_urls_list = sort_urls_by_high_and_medium_score_tokens(search_results)

    # Sort all results by score
    sorted_results = sort_search_results_by_url_and_category(
        search_results, sorted_urls_list)
    total_tokens = sum(result["metadata"].get("num_tokens", 0)
                       for result in sorted_results)

    # Filter search_results directly based on score, MTLD, and link-to-text ratio
    current_tokens = 0
    filtered_results = []
    for result in sorted_results:
        content = f"{result['header']}\n{result['content']}"
        tokens = count_tokens(llm_model, content)
        if current_tokens + tokens > max_tokens:
            break
        filtered_results.append(result)
        current_tokens += tokens

    # Compute filtered_urls based on filtered_results
    filtered_url_stats = defaultdict(
        lambda: {'high_score_tokens': 0, 'medium_score_tokens': 0, 'header_count': 0})
    for result in filtered_results:
        url = result['metadata']['source']
        if result['score'] >= HIGH_QUALITY_SCORE:
            filtered_url_stats[url]['high_score_tokens'] += result['metadata'].get(
                'num_tokens', 0)
            filtered_url_stats[url]['header_count'] += 1
        elif result['score'] >= MEDIUM_QUALITY_SCORE:
            filtered_url_stats[url]['medium_score_tokens'] += result['metadata'].get(
                'num_tokens', 0)
            filtered_url_stats[url]['header_count'] += 1

    # Create filtered_urls list, sorted by high_score_tokens then medium_score_tokens
    filtered_urls = [
        {
            'url': url,
            'high_score_tokens': stats['high_score_tokens'],
            'medium_score_tokens': stats['medium_score_tokens'],
            'header_count': stats['header_count']
        }
        for url, stats in sorted(
            filtered_url_stats.items(),
            key=lambda x: (x[1]['high_score_tokens'],
                           x[1]['medium_score_tokens']),
            reverse=True
        )
    ]

    context = group_results_by_source_for_llm_context(filtered_results)
    prompt = PROMPT_TEMPLATE.format(query=query, context=context)
    response_text = ""
    for chunk in gen.stream_chat(
        prompt,
        llm_model,
        temperature=0.7,
        verbose=True
    ):
        content = chunk["choices"][0]["message"]["content"]
        response_text += content

    input_tokens = count_tokens(llm_model, prompt)
    output_tokens = count_tokens(llm_model, response_text)

    return {
        "query": query,
        "search_engine_results": search_engine_results,
        "started_urls": all_started_urls,
        "searched_urls": all_searched_urls,
        "high_score_urls": create_url_dict_list(all_urls_with_high_scores, search_results),
        "header_docs": header_docs,
        "search_results": search_results,
        "sorted_search_results": sorted_results,
        "filtered_results": filtered_results,
        "filtered_urls": filtered_urls,
        "context": context,
        "response_text": response_text,
        "token_info": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
    }


def web_deep_search_sync(query: str) -> WebDeepSearchResult:
    """
    Synchronous version of web_deep_search that checks for a running event loop.
    
    Args:
        query (str): The search query string.
    
    Returns:
        WebDeepSearchResult: Structured search results.
    
    Raises:
        RuntimeError: If an event loop is already running and cannot be used.
    """
    try:
        # Check if there's a running event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            raise RuntimeError(
                "Cannot run web_deep_search_sync in a running event loop. "
                "Use web_deep_search instead or run in a new thread."
            )
        # If no running loop, run the async function synchronously
        return loop.run_until_complete(web_deep_search(query))
    except RuntimeError as e:
        if "no running event loop" in str(e):
            # Create a new event loop if none exists
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(web_deep_search(query))
                return result
            finally:
                loop.close()
        raise  # Re-raise other RuntimeError exceptions
