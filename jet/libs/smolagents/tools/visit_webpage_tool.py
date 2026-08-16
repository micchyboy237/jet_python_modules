import logging
from dataclasses import dataclass
from pathlib import Path

import requests
from jet.adapters.llama_cpp.chunking_utils import chunk_texts_with_data
from jet.adapters.llama_cpp.config import EMBED_MODEL
from jet.adapters.llama_cpp.hybrid_utils import HybridSearchResult, hybrid_search
from jet.adapters.llama_cpp.model_utils import get_model_ctx_embd_size
from jet.adapters.llama_cpp.token_utils import count_tokens
from jet.adapters.llama_cpp.types import LLAMACPP_EMBED_KEYS
from jet.code.markdown_utils._converters import convert_html_to_markdown
from jet.code.splitter_markdown_utils import get_md_header_contents
from jet.libs.smolagents.utils.debug_saver import DebugSaver
from jet.transformers.object import make_serializable
from jet.utils.inspect_utils import get_entry_file_dir, get_entry_file_name
from smolagents.tools import Tool

logger = logging.getLogger(__name__)


def search_result_serializer(obj):
    if isinstance(obj, dict) and "text" in obj:
        return {
            "text": obj.get("text", ""),
            "score": obj.get("score"),
            "vector_score": obj.get("vector_score"),
        }
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


@dataclass
class PageFetchResult:
    html: str
    success: bool = True
    error_message: str | None = None


def extract_markdown_section_texts(
    md_content: str, ignore_links: bool = True
) -> list[str]:
    """Extract content grouped by headers into markdown-formatted text blocks."""
    header_blocks = get_md_header_contents(md_content, ignore_links=ignore_links)
    return [
        f"{block['header']}\n\n{block['content']}".strip() for block in header_blocks
    ]


class VisitWebpageTool(Tool):
    name = "visit_webpage"
    description = """Visits a webpage and returns the most relevant content excerpts.
Uses hybrid retrieval (vector search + BM25 reranking) to select the top relevant chunks.
This focused output is usually far more useful than returning the entire page.
If you need more context, make a second call with a more specific query."""
    inputs = {
        "url": {"type": "string", "description": "The url of the webpage to visit."},
        "query": {
            "type": "string",
            "description": "(optional) Specific question/topic to focus retrieval on.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(
        self,
        embed_model: LLAMACPP_EMBED_KEYS = EMBED_MODEL,
        max_output_length: int = 3800,
        top_k: int = 5,
        vector_top_k: int = 20,
        chunk_target_tokens: int = 500,
        chunk_overlap_tokens: int = 100,
        min_chunk_tokens: int = 150,  # ← NEW: filter out tiny boilerplate chunks
        verbose: bool = True,
        logs_dir: str | Path | None = None,
    ):
        super().__init__()
        self.embed_model = embed_model
        ctx_embd_size = get_model_ctx_embd_size(self.embed_model)
        self.max_output_tokens = max_output_length
        self.top_k = top_k
        self.vector_top_k = vector_top_k
        self.chunk_target_tokens = chunk_target_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.verbose = verbose
        self.debug_saver = DebugSaver(
            tool_name=self.name,
            base_dir=Path(logs_dir)
            if logs_dir
            else (
                Path(get_entry_file_dir())
                / "generated"
                / Path(get_entry_file_name()).stem
                / "visit_webpage_tool_logs"
            ),
            serializer=search_result_serializer,
        )
        if self.verbose:
            logger.setLevel(logging.DEBUG)

    def _trim_to_token_limit(self, text: str) -> str:
        """Binary search to trim text to max_output_tokens."""
        token_count = count_tokens(text, model=self.embed_model)
        if token_count <= self.max_output_tokens:
            return text
        chars = list(text)
        low, high = 0, len(chars)
        while low < high:
            mid = (low + high + 1) // 2
            candidate = "".join(chars[:mid])
            if (
                count_tokens(candidate, model=self.embed_model)
                <= self.max_output_tokens
            ):
                low = mid
            else:
                high = mid - 1
        return "".join(chars[:low])

    def forward(self, url: str, query: str | None = None) -> str:
        search_query = (
            query.strip()
            if query is not None and isinstance(query, str) and query.strip()
            else "main content and key information from the webpage"
        )
        input_text = f"{url} {search_query}"
        input_tokens = count_tokens(input_text, model=self.embed_model)
        request_data = {
            "url": url,
            "query": query,
            "resolved_search_query": search_query,
            "input_tokens": input_tokens,
        }
        self.debug_saver.save_json("request.json", request_data)
        logger.info("Saved request.json")
        with self.debug_saver.new_call(request_data) as call_dir:
            # Step 1: Fetch page
            fetch_result = self._fetch_url(url)
            if not fetch_result.success:
                error_text = f"Failed to fetch page: {fetch_result.error_message}"
                logger.error(error_text)
                self.debug_saver.save("full_results.md", error_text)
                return error_text
            self.debug_saver.save("page.html", fetch_result.html)
            # Step 2: Extract markdown sections
            md_content = convert_html_to_markdown(fetch_result.html, ignore_links=True)
            self.debug_saver.save("page.md", md_content)

            headings = extract_markdown_section_texts(md_content, ignore_links=True)
            self.debug_saver.save_json("headings.json", headings)
            logger.info(f"Extracted {len(headings)} markdown sections")
            # Step 3: Chunk each section
            chunks = chunk_texts_with_data(
                texts=headings,
                chunk_size=self.chunk_target_tokens,
                chunk_overlap=self.chunk_overlap_tokens,
                strict_sentences=True,
                model=self.embed_model,
                min_chunk_size=self.min_chunk_tokens,  # ← pass through
            )
            self.debug_saver.save_json("chunks.json", chunks)
            logger.info(f"Created {len(chunks)} chunks from {len(headings)} sections")
            # Step 4: Build documents list for hybrid search
            documents = [chunk["content"] for chunk in chunks]
            if not documents:
                msg = "No extractable content found on the page."
                logger.warning(msg)
                self.debug_saver.save("full_results.md", msg)
                return msg
            # Step 5: Hybrid search (vector + BM25)
            if self.verbose:
                logger.info(
                    f"Hybrid search: query='{search_query[:80]}', "
                    f"docs={len(documents)}, top_k={self.top_k}, "
                    f"vector_top_k={self.vector_top_k}"
                )
            results: list[HybridSearchResult] = hybrid_search(
                query=search_query,
                documents=documents,
            )
            self.debug_saver.save_json(
                "search_results.json", make_serializable(results), indent=2
            )
            logger.info(f"Hybrid search returned {len(results)} results")
            # Step 6: Format output with token budget
            header = (
                f"Most relevant excerpts from {url} "
                f"(hybrid vector+BM25 retrieval, query: {search_query!r})\n\n"
            )
            header_tokens = count_tokens(header, model=self.embed_model)
            remaining_tokens = self.max_output_tokens - header_tokens - 100
            if remaining_tokens < 300:
                remaining_tokens = 300
            excerpts = []
            current_tokens = 0
            for rank, r in enumerate(results, 1):
                text = r["text"]
                preview = text[:400] + "..." if len(text) > 400 else text
                line = (
                    f"[{rank}] Score: {r['score']:.3f} "
                    f"(vector: {r['vector_score']:.3f})\n"
                    f"{preview}\n\n"
                )
                line_tokens = count_tokens(line, model=self.embed_model)
                if current_tokens + line_tokens > remaining_tokens:
                    break
                excerpts.append(line)
                current_tokens += line_tokens
            if not excerpts and results:
                # Ensure at least one excerpt
                r = results[0]
                text = r["text"]
                preview = text[:400] + "..." if len(text) > 400 else text
                excerpts = [
                    f"[1] Score: {r['score']:.3f} (vector: {r['vector_score']:.3f})\n"
                    f"{preview}\n\n"
                ]
            result = header + "".join(excerpts)
            self.debug_saver.save("searched_text.md", result)
            logger.info(
                f"Searched text: {len(result)} chars, ~{count_tokens(result, model=self.embed_model)} tokens"
            )
            # Step 7: Trim to token limit
            result = self._trim_to_token_limit(result)
            self.debug_saver.save("trimmed_results.md", result)
            self.debug_saver.save_json(
                "response.json",
                {
                    "output_tokens": count_tokens(result, model=self.embed_model),
                    "char_length": len(result),
                },
                indent=2,
            )
            logger.info(f"Final output: {len(result)} chars")
            return result

    def _fetch_url(self, url: str) -> PageFetchResult:
        try:
            if self.verbose:
                logger.info(f"Fetching URL: {url}")
            resp = requests.get(
                url,
                timeout=18,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; VisitWebpageTool/1.0)"
                },
            )
            resp.raise_for_status()
            return PageFetchResult(html=resp.text)
        except Exception as e:
            msg = f"Fetch failed: {str(e)}"
            if self.verbose:
                logger.error(msg)
            return PageFetchResult(html="", success=False, error_message=msg)


if __name__ == "__main__":
    """
    Demo: VisitWebpageTool standalone test.
    
    Usage:
        python -m jet.libs.smolagents.tools.visit_webpage_tool
        python -m jet.libs.smolagents.tools.visit_webpage_tool --url "https://example.com"
        python -m jet.libs.smolagents.tools.visit_webpage_tool --url "https://example.com" --query "specific topic"
    """
    import argparse

    from rich.console import Console
    from rich.panel import Panel

    console = Console()

    parser = argparse.ArgumentParser(
        description="Test VisitWebpageTool with hybrid search (vector + BM25)"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="https://en.wikipedia.org/wiki/Giant_panda",
        help="Webpage URL to visit (default: Giant Panda Wikipedia)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Optional search query to focus retrieval",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top chunks to return (default: 5)",
    )
    parser.add_argument(
        "--vector-top-k",
        type=int,
        default=20,
        help="Number of vector search candidates for BM25 reranking (default: 20)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=400,
        help="Target tokens per chunk (default: 500)",
    )
    parser.add_argument(
        "--max-output",
        type=int,
        default=3800,
        help="Maximum output tokens (default: 3800)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Enable verbose logging (default: True)",
    )

    args = parser.parse_args()

    # ── Header ──────────────────────────────────────────────
    console.print(
        Panel.fit(
            "[bold cyan]VisitWebpageTool Demo[/bold cyan]\n"
            "[dim]Hybrid Search: Vector Embedding + BM25 Reranking[/dim]",
            border_style="cyan",
        )
    )

    # ── Config Summary ──────────────────────────────────────
    console.print("\n[bold]Configuration:[/bold]")
    console.print(f"  URL:              [yellow]{args.url}[/yellow]")
    console.print(
        f"  Query:            [yellow]{args.query or '(auto: main content)'}[/yellow]"
    )
    console.print(f"  Top-K Chunks:     [green]{args.top_k}[/green]")
    console.print(f"  Vector Top-K:     [green]{args.vector_top_k}[/green]")
    console.print(f"  Chunk Size:       [green]{args.chunk_size} tokens[/green]")
    console.print(f"  Max Output:       [green]{args.max_output} tokens[/green]")

    # ── Instantiate Tool ────────────────────────────────────
    console.print("\n[bold]Initializing tool...[/bold]")
    tool = VisitWebpageTool(
        top_k=args.top_k,
        vector_top_k=args.vector_top_k,
        chunk_target_tokens=args.chunk_size,
        max_output_length=args.max_output,
        verbose=args.verbose,
    )

    # ── Run ─────────────────────────────────────────────────
    console.print("\n[bold]Fetching & searching...[/bold]\n")
    try:
        result = tool.forward(url=args.url, query=args.query)
    except Exception as e:
        console.print(f"[red bold]Error:[/red bold] {e}")
        raise

    # ── Output ──────────────────────────────────────────────
    console.print(
        Panel(
            result,
            title=f"[bold green]Results from {args.url}[/bold green]",
            border_style="green",
            expand=False,
        )
    )

    # ── Stats ───────────────────────────────────────────────
    from jet.adapters.llama_cpp.token_utils import count_tokens

    token_count = count_tokens(result, model=tool.embed_model)
    console.print("\n[bold]Stats:[/bold]")
    console.print(f"  Output chars:     [cyan]{len(result):,}[/cyan]")
    console.print(f"  Output tokens:    [cyan]{token_count:,}[/cyan]")
    console.print(f"  Embed model:      [dim]{tool.embed_model}[/dim]")

    # ── Logs Location ───────────────────────────────────────
    console.print(f"\n[dim]Debug logs saved to: {tool.debug_saver.base_dir}[/dim]")
    console.print("[bold green]✓ Demo complete[/bold green]\n")
