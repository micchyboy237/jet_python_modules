from typing import List, Dict, Optional
from collections import defaultdict
from pydantic import BaseModel, Field
from jet.scrapers.utils import search_data
from jet.servers.mcp.mcp_server import server, Context
from jet.code.markdown_types.markdown_parsed_types import HeaderSearchResult
from jet.vectors.semantic_search.header_vector_search import search_headers
from jet.models.model_types import EmbedModelType
from jet.models.tokenizer.base import count_tokens
from jet.code.html_utils import preprocess_html
from jet.code.markdown_utils._converters import convert_html_to_markdown
from jet.code.markdown_utils._markdown_parser import derive_by_header_hierarchy
from jet.scrapers.hrequests_utils import scrape_urls
from jet.wordnet.analyzers.text_analysis import analyze_readability
from jet.logger import logger


class WebSearchDetailsInput(BaseModel):
    query: str = Field(..., description="Search query string")
    urls_limit: int = Field(
        10, description="Maximum number of URLs to process", ge=1, le=50)
    top_k: Optional[int] = Field(
        None, description="Maximum number of search results", ge=1)
    threshold: float = Field(
        0.0, description="Minimum similarity score for results", ge=0.0, le=1.0)
    chunk_size: int = Field(
        200, description="Size of text chunks", ge=50, le=1000)
    chunk_overlap: int = Field(
        50, description="Overlap between chunks", ge=0, le=200)
    merge_chunks: bool = Field(
        False, description="Merge chunks if False, return individual if True")
    max_tokens: int = Field(
        2000, description="Maximum total tokens for results", ge=500, le=10000)


class WebSearchDetailsOutput(BaseModel):
    results: List[HeaderSearchResult] = Field(
        ..., description="List of search results with metadata")
    urls: List[Dict] = Field(...,
                             description="List of processed URLs with statistics")
    total_tokens: int = Field(..., description="Total token count of results")
    error: Optional[str] = Field(
        None, description="Error message if search failed")


@server.tool(description="Perform a web search and return detailed results with metadata.", annotations={"audience": ["assistant"], "priority": 0.85})
async def web_search_details(arguments: WebSearchDetailsInput, ctx: Context) -> WebSearchDetailsOutput:
    await ctx.info(f"Starting web search for query: {arguments.query}")
    try:
        embed_model: EmbedModelType = "all-MiniLM-L6-v2"
        search_engine_results = search_data(arguments.query, use_cache=True)
        urls = [r["url"] for r in search_engine_results][:arguments.urls_limit]
        header_docs = []
        search_results = []
        headers_total_tokens = 0
        headers_high_score_tokens = 0
        headers_medium_score_tokens = 0
        all_completed_urls = []
        HIGH_QUALITY_SCORE = 0.6
        MEDIUM_QUALITY_SCORE = 0.4
        TARGET_HIGH_SCORE_TOKENS = 4000
        TARGET_MEDIUM_SCORE_TOKENS = 10000

        async for url, status, html in scrape_urls(urls, show_progress=True):
            if status == "completed" and html:
                all_completed_urls.append(url)
                doc_markdown = convert_html_to_markdown(
                    preprocess_html(html), ignore_links=False)
                original_docs = derive_by_header_hierarchy(doc_markdown)
                for doc in original_docs:
                    doc["source"] = url
                sub_results = list(
                    search_headers(
                        original_docs,
                        arguments.query,
                        top_k=arguments.top_k,
                        threshold=arguments.threshold,
                        embed_model=embed_model,
                        chunk_size=arguments.chunk_size,
                        chunk_overlap=arguments.chunk_overlap,
                        tokenizer_model=embed_model,
                        merge_chunks=arguments.merge_chunks
                    )
                )
                filtered_sub_results = []
                for result in sub_results:
                    mtld_result = analyze_readability(result["content"])
                    result["metadata"]["mtld"] = mtld_result["mtld"]
                    result["metadata"]["mtld_category"] = mtld_result["mtld_category"]
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
                    if MEDIUM_QUALITY_SCORE <= result["score"] < HIGH_QUALITY_SCORE
                )
                header_docs.extend(original_docs)
                search_results.extend(filtered_sub_results)
                headers_total_tokens += sub_total_tokens
                headers_high_score_tokens += sub_high_score_tokens
                headers_medium_score_tokens += sub_medium_score_tokens
                if headers_high_score_tokens >= TARGET_HIGH_SCORE_TOKENS or \
                   (headers_high_score_tokens + headers_medium_score_tokens) >= TARGET_MEDIUM_SCORE_TOKENS:
                    logger.info(
                        f"Stopping processing: {headers_high_score_tokens} high-score tokens "
                        f"and {headers_medium_score_tokens} medium-score tokens collected.")
                    break

        search_results = sorted(
            search_results, key=lambda x: x["score"], reverse=True)
        for i, result in enumerate(search_results, 1):
            result["rank"] = i

        filtered_results = []
        current_tokens = 0
        for result in search_results:
            content = f"{result['header']}\n{result['content']}"
            tokens = count_tokens(embed_model, content)
            if current_tokens + tokens > arguments.max_tokens:
                break
            filtered_results.append(result)
            current_tokens += tokens

        url_stats = defaultdict(lambda: {
            "high_score_tokens": 0,
            "medium_score_tokens": 0,
            "header_count": 0,
            "max_score": float('-inf'),
            "min_score": float('inf')
        })
        for result in filtered_results:
            url = result["metadata"]["source"]
            score = result["score"]
            url_stats[url]["header_count"] += 1
            url_stats[url]["max_score"] = max(
                url_stats[url]["max_score"], score)
            url_stats[url]["min_score"] = min(
                url_stats[url]["min_score"], score)
            if result["score"] >= HIGH_QUALITY_SCORE:
                url_stats[url]["high_score_tokens"] += result["metadata"].get(
                    "num_tokens", 0)
            elif result["score"] >= MEDIUM_QUALITY_SCORE:
                url_stats[url]["medium_score_tokens"] += result["metadata"].get(
                    "num_tokens", 0)

        sorted_urls = [
            {
                "url": url,
                "high_score_tokens": stats["high_score_tokens"],
                "medium_score_tokens": stats["medium_score_tokens"],
                "header_count": stats["header_count"],
                "max_score": stats["max_score"],
                "min_score": stats["min_score"]
            }
            for url, stats in sorted(
                url_stats.items(),
                key=lambda x: (x[1]["high_score_tokens"],
                               x[1]["medium_score_tokens"]),
                reverse=True
            )
            if stats["high_score_tokens"] > 0 or stats["medium_score_tokens"] > 0
        ]

        await ctx.report_progress(100, 100, f"Found {len(filtered_results)} matching results")
        return WebSearchDetailsOutput(
            results=filtered_results,
            urls=sorted_urls,
            total_tokens=current_tokens
        )
    except Exception as e:
        await ctx.error(f"Error in web search: {str(e)}")
        return WebSearchDetailsOutput(
            results=[],
            urls=[],
            total_tokens=0,
            error=f"Error in web search: {str(e)}"
        )
