# jet_python_modules/jet/libs/smolagents/agents/web_search_html_info_multi_agent.py
import logging

from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.smolagents.factory import create_code_agent, create_llm_model
from jet.libs.smolagents.agents.html_summary_multi_agent import (
    ScalableHTMLMultiAgentSummarizer,
)
from jet.libs.smolagents.tools.searxng_search_tool import SearXNGSearchTool
from rich.console import Console
from rich.panel import Panel
from smolagents import Tool, ToolCallingAgent

console = Console()
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────
#               Instructions
# ────────────────────────────────────────────────
MANAGER_INSTRUCTIONS = """
You are an intelligent research coordinator managing a team of specialized agents to answer user queries using web information.

## Your Team (Managed Agents)

1. **SearchPlanner** - Analyzes search results and selects the 3-6 most relevant URLs
2. **PageFetcher** - Fetches raw HTML content from provided URLs
3. **HTMLSummarizer** - Converts raw HTML into clean, structured hierarchical summaries

## Your Direct Tool

- **web_search** - Performs web searches to find relevant pages (use this directly, not through an agent)

## Workflow

Follow this exact sequence for every query:

### Step 1: Web Search
- Use the `web_search` tool directly to find relevant pages for the user's query
- Search with descriptive, well-formulated queries

### Step 2: URL Selection
- Pass the raw search results to the **SearchPlanner** agent
- Ask it to analyze and select the most promising URLs that likely contain substantial information
- Specify how many URLs you want (typically 3-6 based on query complexity)

### Step 3: Fetch HTML
- Send the selected URLs to the **PageFetcher** agent
- Request it to fetch the HTML content from ALL selected URLs
- You can send multiple URLs in a single request

### Step 4: Summarize Content
- For EACH fetched page's HTML, ask the **HTMLSummarizer** to produce a structured summary
- Process pages one at a time or in batches, depending on your judgment
- Use the summarizer's `summarize_page` method with the HTML content and URL

### Step 5: Synthesize Final Answer
- Review all summaries collected
- Synthesize a comprehensive, well-structured final answer
- Include relevant citations with URLs
- Address the original query completely and concisely

## Important Rules

- **ALWAYS follow the sequence**: Search → Plan → Fetch → Summarize → Synthesize
- **NEVER skip the SearchPlanner**: Always have it evaluate search results before fetching
- **Delegate appropriately**: Use agents for their specialized tasks, don't try to do their work yourself
- **Handle errors gracefully**: If a page fails to fetch or summarize, note it and continue with available information
- **Cite sources**: Always include URLs for key facts and claims in the final answer
- **Be thorough but efficient**: Don't fetch more pages than necessary, but ensure comprehensive coverage
- **Respect the max_pages limit**: Don't exceed the specified maximum number of pages to fetch

## Response Format

Your final answer should be:
- Well-structured with clear sections if appropriate
- Factual and based only on the summarized content
- Include inline citations or a sources section with URLs
- Concise but comprehensive enough to fully answer the query
"""

# ────────────────────────────────────────────────
#               Specialized Sub-Agents
# ────────────────────────────────────────────────


class SearchPlannerAgent(ToolCallingAgent):
    """Lightweight agent that decides which URLs from search results are worth fetching."""

    def __init__(self, model=None):
        if model is None:
            model = create_llm_model(agent_name="search_planner")

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
            model = create_llm_model(agent_name="page_fetcher")

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
            model = create_llm_model(agent_name="html_summarizer")

        self.summarizer = summarizer or ScalableHTMLMultiAgentSummarizer(
            model_id=LLM_MODEL
        )

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
        model_id: str | None = LLM_MODEL,
        max_pages: int = 5,
    ):
        model = create_llm_model(agent_name="web_orchestrator", model_id=model_id)

        self.search_tool = search_tool or SearXNGSearchTool(max_results=12)

        # Sub-agents
        self.planner = SearchPlannerAgent(model=model)
        self.fetcher = PageFetcherAgent(model=model)
        self.summarizer = HTMLSummarizerAgent(model=model)

        # Main orchestrator with managed agents
        self.manager = create_code_agent(
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
            instructions=MANAGER_INSTRUCTIONS,
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
    model_id: str | None = LLM_MODEL,
    max_pages: int = 5,
) -> WebSearchHTMLInfoMultiAgent:
    return WebSearchHTMLInfoMultiAgent(
        search_tool=search_tool,
        model_id=model_id,
        max_pages=max_pages,
    )
