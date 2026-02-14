# jet_python_modules/jet/libs/smolagents/agents/web_search_html_info_multi_agent.py
import logging

from jet.libs.smolagents.agents.html_summary_multi_agent import (
    ScalableHTMLMultiAgentSummarizer,
)
from jet.libs.smolagents.tools.searxng_search_tool import SearXNGSearchTool
from jet.libs.smolagents.utils.model_utils import create_local_model
from rich.console import Console
from rich.panel import Panel
from smolagents import CodeAgent, Tool, ToolCallingAgent

console = Console()
logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────
#               Specialized Sub-Agents
# ────────────────────────────────────────────────


class SearchPlannerAgent(ToolCallingAgent):
    """Lightweight agent that decides which URLs from search results are worth fetching."""

    def __init__(self, model=None):
        if model is None:
            model = create_local_model(agent_name="search_planner")

        super().__init__(
            tools=[],
            model=model,
            name="SearchPlanner",
            description=(
                "Analyzes search results and selects the 3–6 most relevant URLs "
                "that are likely to contain substantial, high-quality information "
                "about the query. Returns a list of URLs with short justification."
            ),
        )


class FetchHtmlTool(Tool):
    """Dedicated tool class for fetching HTML — defined with class attributes."""

    name = "fetch_html"
    description = (
        "Fetches the raw HTML content from a given webpage URL. "
        "Input must be a complete, valid URL (including https:// or http://)."
    )
    inputs = {
        "url": {"type": "string", "description": "Full URL of the webpage to fetch"}
    }
    output_type = "string"

    def forward(self, url: str) -> str:
        # ← Put real fetching logic here later (requests.get, httpx, etc.)
        # For now keep simulation
        import time

        time.sleep(0.4)  # fake network delay
        return f"[SIMULATED HTML content fetched from {url} at {time.strftime('%Y-%m-%d %H:%M:%S')}]"


class PageFetcherAgent(ToolCallingAgent):
    """Agent responsible for fetching HTML content from URLs."""

    def __init__(self, model=None):
        if model is None:
            model = create_local_model(agent_name="page_fetcher")

        # Now use the proper Tool subclass
        fetch_tool = FetchHtmlTool()

        super().__init__(
            tools=[fetch_tool],
            model=model,
            name="PageFetcher",
            description=(
                "Fetches the full HTML content of selected web pages. "
                "Returns raw HTML or cleaned content ready for DOM parsing."
            ),
        )


class HTMLSummarizerAgent(ToolCallingAgent):
    """Wrapper that uses the existing scalable HTML summarizer."""

    def __init__(
        self, model=None, summarizer: ScalableHTMLMultiAgentSummarizer | None = None
    ):
        if model is None:
            model = create_local_model(agent_name="html_summarizer")

        self.summarizer = summarizer or ScalableHTMLMultiAgentSummarizer(model_id=None)

        super().__init__(
            tools=[],
            model=model,
            name="HTMLSummarizer",
            description=(
                "Takes raw HTML and produces a clean, structured, hierarchical summary "
                "while preserving important facts, tables, links and hierarchy."
            ),
        )

    def summarize_page(self, html: str, url: str) -> str:
        logger.info(f"Summarizing page: {url}")
        summary = self.summarizer.summarize(html)
        return f"URL: {url}\n\n{summary}"


# ────────────────────────────────────────────────
#               Main Orchestrator
# ────────────────────────────────────────────────


class WebSearchHTMLInfoMultiAgent:
    """
    Multi-agent system that:
      1. Searches the web
      2. Selects promising pages
      3. Fetches HTML
      4. Summarizes content hierarchically
      5. Merges insights into final coherent answer
    """

    def __init__(
        self,
        search_tool: Tool | None = None,
        model_id: str | None = None,
        max_pages: int = 5,
    ):
        model = create_local_model(agent_name="web_orchestrator", model_id=model_id)

        self.search_tool = search_tool or SearXNGSearchTool(max_results=12)

        # Sub-agents
        self.planner = SearchPlannerAgent(model=model)
        self.fetcher = PageFetcherAgent(model=model)
        self.summarizer = HTMLSummarizerAgent(model=model)

        # Main orchestrator with managed agents
        self.manager = CodeAgent(
            tools=[self.search_tool],  # ← gives direct access to web search
            model=model,
            managed_agents=[
                self.planner,
                self.fetcher,
                self.summarizer,
            ],
            name="WebHTMLInfoOrchestrator",
            description=(
                "Coordinates web search → page selection → fetching → "
                "hierarchical summarization pipeline to answer complex queries."
            ),
        )
        self.max_pages = max_pages

    def run(self, query: str) -> str:
        logger.info(f"[bold cyan]Starting Web→HTML→Summary pipeline for:[/] {query}")

        initial_prompt = f"""You are an intelligent research coordinator.
Your goal is to answer the user's query using reliable web information.

Query: {query}

Follow this high-level plan:
1. Use the 'web_search' tool to find relevant pages.
2. Ask the SearchPlanner to select the {self.max_pages} most promising URLs.
3. Instruct PageFetcher to retrieve HTML from those URLs.
4. Send each HTML to HTMLSummarizer to get structured summaries.
5. Read all summaries and synthesize a final coherent, well-structured answer.
   Cite sources with URLs where appropriate.

Start by performing a web search.
Be concise but comprehensive in the final answer.
"""

        final_answer = self.manager.run(initial_prompt)

        console.print(
            Panel(
                final_answer.strip(),
                title=f"[bold green]Final Answer – {query[:60]}...[/bold green]",
                border_style="green",
            )
        )
        return final_answer


# Convenience factory / entry point
def create_web_html_info_agent(
    search_tool: Tool | None = None,
    model_id: str | None = None,
    max_pages: int = 5,
) -> WebSearchHTMLInfoMultiAgent:
    return WebSearchHTMLInfoMultiAgent(
        search_tool=search_tool,
        model_id=model_id,
        max_pages=max_pages,
    )
