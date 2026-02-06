# web_search.py

import asyncio
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, TypedDict

from jet.adapters.llama_cpp.llm import LlamacppLLM
from jet.adapters.llama_cpp.tokens import count_tokens, get_tokenizer_fn
from jet.adapters.llama_cpp.types import LLAMACPP_EMBED_KEYS, LLAMACPP_LLM_KEYS
from jet.code.html_utils import preprocess_html
from jet.code.markdown_types.markdown_parsed_types import HeaderDoc, HeaderSearchResult
from jet.code.markdown_utils._converters import convert_html_to_markdown
from jet.code.markdown_utils._markdown_analyzer import analyze_markdown
from jet.code.markdown_utils._markdown_parser import (
    base_parse_markdown,
    derive_by_header_hierarchy,
)
from jet.code.markdown_utils._preprocessors import link_to_text_ratio
from jet.file.utils import save_file
from jet.logger import logger
from jet.scrapers.hrequests_utils import ScrapeStatus, scrape_urls
from jet.scrapers.utils import scrape_links, search_data
from jet.vectors.semantic_search.header_vector_search import search_headers
from jet.wordnet.analyzers.text_analysis import calculate_mtld, calculate_mtld_category

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "generated",
    os.path.splitext(os.path.basename(__file__))[0],
)

PROMPT_TEMPLATE = """\
Context information is below.
---------------------
{context}
---------------------

Given the context information, answer the query.

Query: {query}
"""

SYSTEM_MESSAGE = """\
You are a precise research assistant. Answer using **only** the provided context.

Always structure your response exactly like this:

**Answer**
[One clear paragraph or 1–2 sentences giving the direct answer]

**Key Facts**
• Fact / number / date / name 1
• Fact 2
• …

**Explanation** (only if needed)
[Additional context, quotes, reasoning — keep short]

**Sources** (list only if multiple different origins)
• https://... (brief note)
• ...

Rules:
• If partial info only → say so in **Answer** ("According to partial information…", "Limited data suggests…")
• If nothing relevant → only write: **Answer**  
  No relevant information found in the provided context.
• Ignore ads, menus, footers, boilerplate
• Use bullets, never long paragraphs without breaks
"""

HIGH_QUALITY_SCORE = 0.6
MEDIUM_QUALITY_SCORE = 0.4
TARGET_HIGH_SCORE_TOKENS = 4000
TARGET_MEDIUM_SCORE_TOKENS = 10000


def sort_urls_by_high_and_medium_score_tokens(
    results: list[HeaderSearchResult],
) -> list[str]:
    # Group results by URL and calculate total medium_score_tokens per URL
    url_medium_score_tokens = defaultdict(int)
    for result in results:
        url = result["metadata"].get("source", "Unknown")
        if (
            result["score"] >= MEDIUM_QUALITY_SCORE
            and result.get("mtld_category") != "very_low"
        ):
            url_medium_score_tokens[url] += result["metadata"].get("num_tokens", 0)

    # Calculate high and medium score tokens per URL
    url_score_tokens = defaultdict(
        lambda: {"high_score_tokens": 0, "medium_score_tokens": 0}
    )

    sorted_urls = sorted(
        url_score_tokens.keys(),
        key=lambda url: (
            url_score_tokens[url]["high_score_tokens"],
            url_score_tokens[url]["medium_score_tokens"],
        ),
        reverse=True,
    )

    return sorted_urls


def sort_search_results_by_url_and_category(
    results: list[HeaderSearchResult], sorted_urls: list[str]
):
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
        url_group_sorted = sorted(url_group, key=lambda r: r["score"], reverse=True)
        sorted_high_score.extend(url_group_sorted)
    # Add any >= HIGH_QUALITY_SCORE results whose url wasn't in sorted_urls, sorted by score
    if high_score_results:
        sorted_high_score.extend(
            sorted(high_score_results, key=lambda r: r["score"], reverse=True)
        )

    # Stage 2: Medium score results (MEDIUM_QUALITY_SCORE <= score < HIGH_QUALITY_SCORE), sort by score descending
    sorted_medium_score = sorted(
        medium_score_results, key=lambda r: r["score"], reverse=True
    )

    # Stage 3: Low score results (score < MEDIUM_QUALITY_SCORE), sort by score descending
    sorted_low_score = sorted(low_score_results, key=lambda r: r["score"], reverse=True)

    return sorted_high_score + sorted_medium_score + sorted_low_score


def group_results_by_source_for_llm_context(
    results: list[HeaderSearchResult],
    llm_model: LLAMACPP_LLM_KEYS = "qwen3-instruct-2507:4b",
) -> str:
    def strip_hashtags(text: str) -> str:
        if text:
            return text.lstrip("#").strip()
        return text

    # Initialize tokenizer
    tokenize = get_tokenizer_fn(llm_model, add_special_tokens=False)
    separator = "\n\n"
    separator_tokens = len(tokenize(separator))

    # Calculate high and medium score tokens per URL
    url_score_tokens = defaultdict(
        lambda: {"high_score_tokens": 0, "medium_score_tokens": 0}
    )
    for result in results:
        url = result["metadata"].get("source", "Unknown")
        if result["score"] >= HIGH_QUALITY_SCORE:
            url_score_tokens[url]["high_score_tokens"] += result["metadata"].get(
                "num_tokens", 0
            )
        elif result["score"] >= MEDIUM_QUALITY_SCORE:
            url_score_tokens[url]["medium_score_tokens"] += result["metadata"].get(
                "num_tokens", 0
            )

    # Sort URLs by high_score_tokens, then medium_score_tokens (descending)
    sorted_urls = sorted(
        url_score_tokens.keys(),
        key=lambda url: (
            url_score_tokens[url]["high_score_tokens"],
            url_score_tokens[url]["medium_score_tokens"],
        ),
        reverse=True,
    )

    # Group results by URL and sort within each URL by score
    grouped_temp: defaultdict[str, list[HeaderSearchResult]] = defaultdict(list)
    seen_header_text: defaultdict[str, set[str]] = defaultdict(set)
    for result in results:
        url = result["metadata"].get("source", "Unknown")
        grouped_temp[url].append(result)

    context_blocks = []
    for url in sorted_urls:
        docs = sorted(grouped_temp[url], key=lambda x: x["score"], reverse=True)
        block = f"<!-- Source: {url} -->\n\n"
        seen_header_text_in_block = set()

        # Group by doc_index and header to handle overlaps
        grouped_by_header: defaultdict[tuple[int, str], list[HeaderSearchResult]] = (
            defaultdict(list)
        )
        for doc in sorted(
            docs,
            key=lambda x: (
                x["metadata"].get("doc_index", 0),
                x["metadata"].get("start_idx", 0),
            ),
        ):
            doc_index = doc["metadata"].get("doc_index", 0)
            header = doc.get("header", "") or ""
            grouped_by_header[(doc_index, header)].append(doc)

        for (doc_index, header), chunks in grouped_by_header.items():
            parent_header = chunks[0].get("parent_header", "None")
            parent_level = chunks[0]["metadata"].get("parent_level", None)
            doc_level = (
                chunks[0]["metadata"].get("level", 0)
                if chunks[0]["metadata"].get("level") is not None
                else 0
            )
            parent_header_key = (
                strip_hashtags(parent_header)
                if parent_header and parent_header != "None"
                else None
            )
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
            if (
                parent_header_key
                and parent_level is not None
                and has_matching_child
                and parent_header_key not in seen_header_text_in_block
            ):
                block += f"{parent_header}\n\n"
                seen_header_text_in_block.add(parent_header_key)

            # Add header if it hasn't been added
            if (
                header_key
                and header_key not in seen_header_text_in_block
                and doc_level >= 0
            ):
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
                        f"Non-string content in chunk for source: {url}, doc_index: {doc_index}, type: {type(next_content)}. Converting to string."
                    )
                    next_content = str(next_content) if next_content else ""

                # Merge if chunks overlap or are adjacent
                if next_start <= end_idx + 1:
                    overlap = end_idx - next_start + 1 if next_start <= end_idx else 0
                    additional_content = (
                        next_content[overlap:] if overlap > 0 else next_content
                    )
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

        block_tokens = len(tokenize(block))
        if block_tokens > len(tokenize(f"<!-- Source: {url} -->\n\n")):
            context_blocks.append(block.strip())
        else:
            logger.warning(f"Empty block for {url} after processing; skipping.")

    result = "\n\n".join(context_blocks)
    final_token_count = len(tokenize(result))
    logger.debug(
        f"Grouped context created with {final_token_count} tokens for {len(grouped_temp)} sources"
    )
    return result


# Helper function to create list of dicts for URLs
def create_url_dict_list(
    urls: list[str], search_results: list[HeaderSearchResult]
) -> list[dict]:
    # Calculate stats per URL from search_results
    url_stats = defaultdict(
        lambda: {
            "high_score_tokens": 0,
            "high_score_headers": 0,
            "medium_score_tokens": 0,
            "medium_score_headers": 0,
            "headers": 0,
            "max_score": float("-inf"),
            "min_score": float("inf"),
        }
    )
    for result in search_results:
        url = result["metadata"].get("source", "Unknown")
        score = result["score"]
        url_stats[url]["headers"] += 1
        url_stats[url]["max_score"] = max(url_stats[url]["max_score"], score)
        url_stats[url]["min_score"] = min(url_stats[url]["min_score"], score)
        if result["score"] >= HIGH_QUALITY_SCORE:
            url_stats[url]["high_score_tokens"] += result["metadata"].get(
                "num_tokens", 0
            )
            url_stats[url]["high_score_headers"] += 1
        elif result["score"] >= MEDIUM_QUALITY_SCORE:
            url_stats[url]["medium_score_tokens"] += result["metadata"].get(
                "num_tokens", 0
            )
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


class SearchStats(TypedDict):
    total_tokens: int
    high_score_tokens: int
    medium_score_tokens: int
    mtld_score_average: float
    urls_with_high_scores: list[str]
    urls_with_low_scores: list[str]


class UrlTokenStat(TypedDict):
    url: str
    high_score_tokens: int
    medium_score_tokens: int
    header_count: int


class HtmlStatusItem(TypedDict):
    status: ScrapeStatus
    html: str


HtmlStatus = dict[str, HtmlStatusItem]


class HybridSearchResult(TypedDict):
    query: str
    search_engine_results: list[dict]
    collected_urls: list[str]
    header_docs: list[HeaderDoc]
    all_search_results: list[HeaderSearchResult]  # before final filtering
    filtered_results: list[HeaderSearchResult]  # after token budget
    grouped_context: str
    llm_messages: list[dict]
    llm_response: str
    token_counts: dict[Literal["input", "output", "total"], int]
    stats: SearchStats
    url_stats: list[UrlTokenStat]
    settings: dict[str, Any]
    all_htmls_with_status: HtmlStatus


async def hybrid_search(
    query: str,
    *,
    use_cache: bool = False,
    llm_log_dir: Path | str | None = None,
    embed_model: LLAMACPP_EMBED_KEYS = "nomic-embed-text",
    llm_model: LLAMACPP_LLM_KEYS = "qwen3-instruct-2507:4b",
    max_tokens: int = 4000,
    urls_limit: int = 10,
    top_k: int | None = None,
    threshold: float = 0.0,
    chunk_size: int = 200,
    chunk_overlap: int = 50,
    merge_chunks: bool = False,
    early_stop: bool = False,
) -> HybridSearchResult:
    """Perform hybrid search and return structured results without side-effects."""

    search_engine_results = search_data(query, use_cache=use_cache)

    urls = [r["url"] for r in search_engine_results][:urls_limit]

    html_list: list[str] = []
    header_docs: list[HeaderDoc] = []
    search_results: list[HeaderSearchResult] = []

    headers_total_tokens = 0
    headers_high_score_tokens = 0
    headers_medium_score_tokens = 0
    headers_mtld_scores: list[float] = []

    all_started_urls = []
    all_completed_urls = []
    all_searched_urls = []
    all_urls_with_high_scores = []
    all_urls_with_low_scores = []
    all_htmls_with_status: HtmlStatus = {}

    async for url, status, html in scrape_urls(urls, show_progress=True):
        if html:
            all_htmls_with_status[url] = {
                "status": status,
                "html": html,
            }

        if status == "started":
            all_started_urls.append(url)
            continue

        if status != "completed" or not html:
            continue

        all_completed_urls.append(url)
        html_list.append(html)
        all_searched_urls.append(url)

        # ── Per-page saving removed ───────────────────────────────────────
        # sub_source_dir = ...
        # sub_output_dir = ...
        # save_file(html, ...)
        # save_file(preprocess_html(html), ...)
        # save_file(links, ...)
        # save_file(doc_markdown, ...)
        # save_file(doc_analysis, ...)
        # save_file(doc_markdown_tokens, ...)
        # save_file(original_docs, ...)

        original_docs: list[HeaderDoc] = derive_by_header_hierarchy(
            convert_html_to_markdown(html, ignore_links=True), ignore_links=True
        )

        for doc in original_docs:
            doc["source"] = url  # type: ignore

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
                merge_chunks=merge_chunks,
            )
        )

        filtered_sub_results = []
        for result in sub_results:
            ltr = link_to_text_ratio(result["content"])
            mtld = calculate_mtld(result["content"])
            mtld_cat = calculate_mtld_category(mtld)

            result["metadata"]["ltr_ratio"] = ltr
            result["metadata"]["mtld"] = mtld
            result["metadata"]["mtld_category"] = mtld_cat

            if result["score"] >= MEDIUM_QUALITY_SCORE and mtld_cat != "very_low":
                filtered_sub_results.append(result)

        sub_total_tokens = sum(
            r["metadata"].get("num_tokens", 0) for r in filtered_sub_results
        )
        sub_high = sum(
            r["metadata"].get("num_tokens", 0)
            for r in filtered_sub_results
            if r["score"] >= HIGH_QUALITY_SCORE
        )
        sub_medium = sum(
            r["metadata"].get("num_tokens", 0)
            for r in filtered_sub_results
            if MEDIUM_QUALITY_SCORE <= r["score"] < HIGH_QUALITY_SCORE
        )

        sub_mtld_values = [
            r["metadata"]["mtld"]
            for r in filtered_sub_results
            if r["score"] >= HIGH_QUALITY_SCORE
        ]
        sub_mtld_avg = (
            sum(sub_mtld_values) / len(sub_mtld_values) if sub_mtld_values else 0.0
        )

        header_docs.extend(original_docs)
        search_results.extend(filtered_sub_results)

        if sub_high > 0:
            all_urls_with_high_scores.append(url)
        else:
            all_urls_with_low_scores.append(url)

        headers_total_tokens += sub_total_tokens
        headers_high_score_tokens += sub_high
        headers_medium_score_tokens += sub_medium
        headers_mtld_scores.append(sub_mtld_avg)

        if (
            early_stop
            and headers_high_score_tokens >= TARGET_HIGH_SCORE_TOKENS
            or (headers_high_score_tokens + headers_medium_score_tokens)
            >= TARGET_MEDIUM_SCORE_TOKENS
        ):
            logger.info(f"Early stop after {url} – token targets reached")
            break

    # ── Final aggregation & sorting ──────────────────────────────────────────

    # Sort & rank
    search_results = sorted(search_results, key=lambda x: x["score"], reverse=True)
    for i, r in enumerate(search_results, 1):
        r["rank"] = i

    # URL stats
    url_stats = defaultdict(lambda: {"high": 0, "medium": 0, "count": 0})
    for r in search_results:
        src = r["metadata"].get("source", "unknown")
        tok = r["metadata"].get("num_tokens", 0)
        if r["score"] >= HIGH_QUALITY_SCORE:
            url_stats[src]["high"] += tok
            url_stats[src]["count"] += 1
        elif r["score"] >= MEDIUM_QUALITY_SCORE:
            url_stats[src]["medium"] += tok
            url_stats[src]["count"] += 1

    sorted_url_stats: list[UrlTokenStat] = sorted(
        (
            {
                "url": url,
                "high_score_tokens": s["high"],
                "medium_score_tokens": s["medium"],
                "header_count": s["count"],
            }
            for url, s in url_stats.items()
            if s["high"] > 0 or s["medium"] > 0
        ),
        key=lambda x: (x["high_score_tokens"], x["medium_score_tokens"]),
        reverse=True,
    )

    # Final token-limited filtering
    current_tokens = 0
    filtered_results = []
    for result in search_results:  # already sorted by score
        content = f"{result['header']}\n{result['content']}"
        tokens = count_tokens(content, llm_model)
        if current_tokens + tokens > max_tokens:
            break
        filtered_results.append(result)
        current_tokens += tokens

    # Build final context
    grouped_context = group_results_by_source_for_llm_context(
        filtered_results, llm_model
    )

    # LLM call
    llm = LlamacppLLM(
        model=llm_model,
        base_url="http://shawn-pc.local:8080/v1",
        verbose=True,
        log_dir=str(llm_log_dir) if llm_log_dir else None,
    )
    prompt = PROMPT_TEMPLATE.format(query=query, context=grouped_context)
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": prompt},
    ]
    llm_response_stream = llm.chat(messages, temperature=0.3, stream=True)
    llm_response = ""
    for chunk in llm_response_stream:
        llm_response += chunk

    input_tokens = count_tokens(prompt, llm_model)
    output_tokens = count_tokens(llm_response, llm_model)

    return {
        "query": query,
        "search_engine_results": search_engine_results,
        "collected_urls": all_completed_urls,
        "header_docs": header_docs,
        "all_search_results": search_results,
        "filtered_results": filtered_results,
        "grouped_context": grouped_context,
        "llm_messages": messages,
        "llm_response": llm_response,
        "token_counts": {
            "input": input_tokens,
            "output": output_tokens,
            "total": input_tokens + output_tokens,
        },
        "stats": {
            "total_tokens": headers_total_tokens,
            "high_score_tokens": headers_high_score_tokens,
            "medium_score_tokens": headers_medium_score_tokens,
            "mtld_score_average": round(
                sum(headers_mtld_scores) / len(headers_mtld_scores), 2
            )
            if headers_mtld_scores
            else 0.0,
            "urls_with_high_scores": all_urls_with_high_scores,
            "urls_with_low_scores": all_urls_with_low_scores,
        },
        "url_stats": sorted_url_stats,
        "settings": {
            "urls_limit": urls_limit,
            "embed_model": embed_model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "max_tokens": max_tokens,
        },
        "all_htmls_with_status": all_htmls_with_status,
    }


if __name__ == "__main__":
    import argparse
    import asyncio
    import shutil

    from jet.code.html_utils import preprocess_html
    from jet.code.markdown_utils._converters import convert_html_to_markdown
    from jet.code.markdown_utils._markdown_analyzer import analyze_markdown
    from jet.code.markdown_utils._markdown_parser import base_parse_markdown
    from jet.file.utils import save_file
    from jet.scrapers.utils import scrape_links
    from jet.utils.text import format_sub_dir, format_sub_source_dir

    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default="Top isekai anime 2026")
    args = parser.parse_args()

    use_cache = True
    early_stop = True

    query_output_dir = f"{OUTPUT_DIR}/{format_sub_dir(args.query)}"
    shutil.rmtree(query_output_dir, ignore_errors=True)

    llm_log_dir = Path(query_output_dir) / "llm_calls"

    result = asyncio.run(
        hybrid_search(
            args.query,
            llm_log_dir=llm_log_dir,
            use_cache=use_cache,
            early_stop=early_stop,
        )
    )

    print(f"Found {len(result['filtered_results'])} relevant chunks")
    print(f"LLM response length: {len(result['llm_response'])} chars")

    search_engine_results = result.pop("search_engine_results")
    header_docs = result.pop("header_docs")
    all_search_results = result.pop("all_search_results")
    filtered_results = result.pop("filtered_results")
    url_stats = result.pop("url_stats")
    stats = result.pop("stats")
    grouped_context = result.pop("grouped_context")
    llm_messages = result.pop("llm_messages")
    llm_response = result.pop("llm_response")
    token_counts = result.pop("token_counts")
    all_htmls_with_status = result.pop("all_htmls_with_status")

    save_file(result, f"{query_output_dir}/result_meta.json")
    save_file(search_engine_results, f"{query_output_dir}/search_engine_results.json")
    save_file(header_docs, f"{query_output_dir}/header_docs.json")
    save_file(all_search_results, f"{query_output_dir}/all_search_results.json")
    save_file(filtered_results, f"{query_output_dir}/filtered_results.json")
    save_file(url_stats, f"{query_output_dir}/url_stats.json")
    save_file(stats, f"{query_output_dir}/stats.json")
    save_file(grouped_context, f"{query_output_dir}/grouped_context.md")
    save_file(llm_messages, f"{query_output_dir}/llm_messages.json")
    save_file(llm_response, f"{query_output_dir}/llm_response.md")
    save_file(token_counts, f"{query_output_dir}/token_counts.json")

    for url, info in all_htmls_with_status.items():
        status = info["status"]
        html = info["html"]

        sub_source_dir = format_sub_source_dir(url)
        sub_output_dir = Path(query_output_dir) / "pages" / sub_source_dir
        save_file(html, f"{sub_output_dir}/page.html")
        save_file(preprocess_html(html), f"{sub_output_dir}/page_preprocessed.html")

        links = set(scrape_links(html, url))
        links = [
            link
            for link in links
            if (link != url if isinstance(link, str) else link["url"] != url)
        ]
        save_file(links, sub_output_dir / "links.json")

        doc_markdown = convert_html_to_markdown(html, ignore_links=False)
        save_file(doc_markdown, f"{sub_output_dir}/page.md")

        doc_analysis = analyze_markdown(doc_markdown)
        save_file(doc_analysis, sub_output_dir / "analysis.json")
        doc_markdown_tokens = base_parse_markdown(doc_markdown)
        save_file(doc_markdown_tokens, sub_output_dir / "markdown_tokens.json")

        original_docs: list[HeaderDoc] = derive_by_header_hierarchy(
            doc_markdown, ignore_links=True
        )

        save_file(original_docs, sub_output_dir / "docs.json")
