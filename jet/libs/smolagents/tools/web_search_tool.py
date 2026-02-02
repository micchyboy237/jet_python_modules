# web_search_tool.py
import logging
from pathlib import Path

import requests
from jet.libs.smolagents.utils.debug_saver import DebugSaver
from jet.utils.inspect_utils import get_entry_file_dir, get_entry_file_name
from smolagents.tools import Tool

logger = logging.getLogger(__name__)


from jet.adapters.llama_cpp.tokens import count_tokens
from jet.adapters.llama_cpp.types import LLAMACPP_EMBED_KEYS


class WebSearchTool(Tool):
    name = "web_search"
    description = "Performs a web search for a query and returns a string of the top search results formatted as markdown with titles, links, and descriptions."
    inputs = {
        "query": {"type": "string", "description": "The search query to perform."},
    }
    output_type = "string"

    def __init__(
        self,
        max_results: int = 10,
        engine: str = "duckduckgo",
        verbose: bool = True,
        logs_dir: str | Path | None = None,
        embed_model: LLAMACPP_EMBED_KEYS = "nomic-embed-text",
    ):
        super().__init__()
        self.max_results = max_results
        self.engine = engine
        self.verbose = verbose
        self.embed_model = embed_model
        _caller_base_dir = (
            Path(get_entry_file_dir())
            / "generated"
            / Path(get_entry_file_name()).stem
            / "web_search_tool_logs"
        )

        self.debug_saver = DebugSaver(
            tool_name=self.name,
            base_dir=Path(logs_dir).resolve() if logs_dir else _caller_base_dir,
        )
        if self.verbose:
            logger.setLevel(logging.DEBUG)

    def forward(self, query: str) -> str:
        input_text_for_tokens = query.strip()
        input_tokens = count_tokens(input_text_for_tokens, model=self.embed_model)

        request_data = {
            "query": query,
            "engine": self.engine,
            "max_results": self.max_results,
            "input_tokens": input_tokens,
        }
        self.debug_saver.save_json("request.json", request_data)
        logger.info("Saved request.json")

        with self.debug_saver.new_call(request_data) as call_dir:
            results = self.search(query)

            if len(results) == 0:
                msg = "No results found! Try a less restrictive/shorter query."
                if call_dir:
                    self.debug_saver.save("full_results.md", msg)
                raise Exception(msg)

            # ── New: save raw search results ────────────────
            self.debug_saver.save_json("raw_results.json", results, indent=2)
            logger.info("Saved raw_results.json")

            formatted = self.parse_results(results)

            if call_dir:
                output_tokens = count_tokens(formatted, model=self.embed_model)
                # Save full formatted result as response.json (more useful)
                self.debug_saver.save_json(
                    "response.json",
                    {
                        "result": formatted,
                        "result_length": len(formatted),
                        "source_result_count": len(results),
                        "output_tokens": output_tokens,
                    },
                    indent=2,
                )
                logger.info("Saved response.json (full formatted result)")

                # Full markdown output — what the agent actually receives
                self.debug_saver.save("full_results.md", formatted)

            return formatted

    def search(self, query: str) -> list:
        if self.engine == "duckduckgo":
            return self.search_duckduckgo(query)
        if self.engine == "bing":
            return self.search_bing(query)
        raise ValueError(f"Unsupported engine: {self.engine}")

    def parse_results(self, results: list) -> str:
        lines = ["## Search Results\n"]
        for r in results:
            lines.append(f"### [{r['title']}]({r['link']})")
            if r.get("description"):
                lines.append("")
                lines.append(r["description"].strip())
            lines.append("")
        return "\n".join(lines)

    # ──────────────────────────────────────────────
    # search_duckduckgo and _create_duckduckgo_parser
    # remain unchanged — just keeping them for completeness
    # ──────────────────────────────────────────────

    def search_duckduckgo(self, query: str) -> list:
        response = requests.get(
            "https://lite.duckduckgo.com/lite/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        parser = self._create_duckduckgo_parser()
        parser.feed(response.text)
        return parser.results

    def _create_duckduckgo_parser(self):
        from html.parser import HTMLParser

        class SimpleResultParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.results = []
                self.current = {}
                self.capture_title = False
                self.capture_description = False
                self.capture_link = False

            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if tag == "a" and attrs.get("class") == "result-link":
                    self.capture_title = True
                elif tag == "td" and attrs.get("class") == "result-snippet":
                    self.capture_description = True
                elif tag == "span" and attrs.get("class") == "link-text":
                    self.capture_link = True

            def handle_endtag(self, tag):
                if tag == "a" and self.capture_title:
                    self.capture_title = False
                elif tag == "td" and self.capture_description:
                    self.capture_description = False
                elif tag == "span" and self.capture_link:
                    self.capture_link = False
                elif tag == "tr":
                    if {"title", "description", "link"} <= self.current.keys():
                        self.current["description"] = " ".join(
                            self.current["description"]
                        )
                        self.results.append(self.current)
                        self.current = {}

            def handle_data(self, data):
                if self.capture_title:
                    self.current["title"] = data.strip()
                elif self.capture_description:
                    self.current.setdefault("description", []).append(data.strip())
                elif self.capture_link:
                    self.current["link"] = "https://" + data.strip()

        return SimpleResultParser()

    def search_bing(self, query: str) -> list:
        import xml.etree.ElementTree as ET

        response = requests.get(
            "https://www.bing.com/search",
            params={"q": query, "format": "rss"},
        )
        response.raise_for_status()
        root = ET.fromstring(response.text)
        items = root.findall(".//item")
        return [
            {
                "title": item.findtext("title") or "No title",
                "link": item.findtext("link") or "",
                "description": item.findtext("description") or "",
            }
            for item in items[: self.max_results]
        ]
