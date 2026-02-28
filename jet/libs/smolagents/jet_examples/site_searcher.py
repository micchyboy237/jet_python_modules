#!/usr/bin/env python3
"""
smolagents Multi-Agent system: Search-any-site-from-URL + summarize results
================================================================================

Requires:
    pip install smolagents selenium helium pillow requests markdownify

Uses Helium + Selenium under the hood (like in web_browser.py example).
"""

import re
from typing import Optional

import helium
from jet.libs.smolagents.utils.model_utils import create_local_model
from markdownify import markdownify
from rich.console import Console
from rich.panel import Panel
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from smolagents import (
    CodeAgent,
    Tool,
    ToolCallingAgent,
)

console = Console()


# ────────────────────────────────────────────────
#  Browser / Interaction Tools  (used by sub-agent)
# ────────────────────────────────────────────────


class GoToTool(Tool):
    name = "go_to"
    description = "Navigate to the specified webpage URL."

    def forward(self, url: str) -> str:
        import helium

        try:
            helium.go_to(url)
            driver = helium.get_driver()
            current_url = getattr(driver, "current_url", None) if driver else None
            if current_url:
                return f"Successfully navigated to: {current_url}"
            else:
                return "Navigation attempted but could not verify current URL."
        except Exception as e:
            return f"Navigation failed: {str(e)}"


class TypeIntoSearchBoxTool(Tool):
    name = "type_into_search_box"
    description = "Type the query into the most likely search input field and submit. Attempts to auto-detect a search input, with an optional custom selector."

    def forward(self, query: str, selector: Optional[str] = None) -> str:
        import helium

        driver = helium.get_driver()
        if driver is None:
            return "Driver is not initialized."
        try:
            # Try multiple common selectors
            search_input = None
            candidate_selectors = [
                "input[type='search']",
                "input[name='q']",
                "input[role='searchbox']",
                "input[placeholder*='Search' i]",
                "input[aria-label*='search' i]",
                "input[id*='search' i]",
            ]
            for sel in candidate_selectors:
                try:
                    elem = driver.find_element(By.CSS_SELECTOR, sel)
                    if elem.is_displayed() and elem.is_enabled():
                        search_input = elem
                        break
                except Exception:
                    continue

            if not search_input:
                # Fallback to provided / custom selector or fallback selector
                sel_to_try = (
                    selector
                    if selector
                    else "input[type='search'], input[name='q'], input[role='searchbox'], input[placeholder*='Search']"
                )
                try:
                    search_input = driver.find_element(By.CSS_SELECTOR, sel_to_try)
                except Exception as e:
                    return f"Could not find search box ({sel_to_try}): {str(e)}"

            search_input.clear()
            search_input.send_keys(query)
            search_input.send_keys(Keys.ENTER)

            return f"Submitted search query: '{query}' → waiting for results..."
        except Exception as e:
            return f"Failed to find or interact with search box: {str(e)}"


class GetPageMarkdownTool(Tool):
    name = "get_page_content_as_markdown"
    description = "Get the current page rendered as clean markdown."

    def forward(self, max_length: int = 8000) -> str:
        import helium

        driver = helium.get_driver()
        if driver is None:
            return "Driver is not initialized."
        try:
            html = getattr(driver, "page_source", None)
            if not html:
                return "Page source unavailable from driver."
            md = markdownify(html, heading_style="ATX")
            md = re.sub(r"\n{3,}", "\n\n", md.strip())
            if len(md) > max_length:
                md = md[:max_length] + "\n\n… (truncated)"
            return md
        except Exception as e:
            return f"Error extracting content: {str(e)}"


class SummarizeSearchResultsTool(Tool):
    name = "summarize_search_results"
    description = "Extract and summarize visible search results from current page."

    def forward(self) -> str:
        import helium

        driver = helium.get_driver()
        if driver is None:
            return "Driver is not initialized."
        try:
            selectors = [
                "div.g",  # Google
                "li.b_algo",  # Bing
                "article.result",  # DuckDuckGo-ish
                "div.result, div.search-result",
                "h3 a",  # fallback titles
            ]
            items = []
            for sel in selectors:
                try:
                    elements = driver.find_elements(By.CSS_SELECTOR, sel)
                    if elements:
                        for el in elements[:8]:
                            text = el.text.strip()
                            if text and len(text) > 30:
                                items.append(text[:300])
                        if items:
                            break
                except Exception:
                    continue

            if not items:
                # Very fallback: just take headings + links
                headings = driver.find_elements(By.CSS_SELECTOR, "h2, h3, h4")
                for h in headings[:10]:
                    txt = h.text.strip()
                    if txt:
                        items.append(txt)

            if items:
                summary = "Top relevant results snippets:\n\n" + "\n───\n".join(items)
                if len(summary) > 2200:
                    summary = summary[:2200] + "\n… (truncated)"
                return summary
            else:
                return "Could not clearly identify search result items on the page."
        except Exception as e:
            return f"Error during result extraction: {str(e)}"


# ────────────────────────────────────────────────
#  Sub-agent: Browser / Search Executor
# ────────────────────────────────────────────────


def create_search_sub_agent(
    max_steps: int = 12, verbosity: int = 1
) -> ToolCallingAgent:
    model = create_local_model(temperature=0.55, agent_name="search_browser_agent")

    return ToolCallingAgent(
        tools=[
            go_to,
            type_into_search_box,
            get_page_content_as_markdown,
            summarize_search_results,
        ],
        model=model,
        max_steps=max_steps,
        name="search_browser_agent",
        description=(
            "Specialized agent that can navigate to a website, find a search box, "
            "type a query, submit it, and summarize the search results. "
            "Use this when you need to perform a search on a specific site given its URL."
        ),
        verbosity_level=verbosity,
    )


# ────────────────────────────────────────────────
#  Manager Agent
# ────────────────────────────────────────────────


def create_search_manager_agent(sub_agents: list, max_steps: int = 8) -> CodeAgent:
    model = create_local_model(temperature=0.65, agent_name="search_manager")

    return CodeAgent(
        tools=[],  # manager delegates, doesn't use tools directly
        model=model,
        managed_agents=sub_agents,
        max_steps=max_steps,
        verbosity_level=1,
        planning_interval=4,  # plan every 4 steps
        additional_authorized_imports=["re", "time"],
    )


# ────────────────────────────────────────────────
#  Main entry point
# ────────────────────────────────────────────────


def run_site_search(url: str, query: str, headless: bool = False) -> str:
    """
    High-level function to run the multi-agent search-on-site system.
    """
    console.rule("Starting Site-Specific Search Agent System")

    # Initialize Helium browser once
    from seleniumbase import Driver

    driver = Driver(uc=True, headless=headless)
    helium.set_driver(driver)

    try:
        sub_agent = create_search_sub_agent(max_steps=14, verbosity=2)
        manager = create_search_manager_agent([sub_agent], max_steps=10)

        task = f"""\
You need to perform a search using this specific website:

URL: {url}
Search query: {query}

Steps you should follow:
1. Navigate to the given URL
2. Find the main search input field and submit the query
3. After results load, extract and summarize the most relevant results
4. Return a concise summary of what the page says about the query

Use the search_browser_agent to do the actual navigation and interaction.
"""

        console.print(
            f"\n[bold cyan]Task sent to manager:[/bold cyan]\n{task.strip()}\n"
        )

        answer = manager.run(task)
        return answer

    finally:
        try:
            helium.kill_browser()
        except:
            pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a site-specific search using a multi-agent search-on-site system."
    )
    parser.add_argument(
        "url",
        type=str,
        help="The target website URL to search",
        nargs="?",
    )
    parser.add_argument(
        "query",
        type=str,
        help="The search query to use on the site",
        nargs="?",
    )
    parser.add_argument(
        "-H",
        "--headless",
        action="store_true",
        help="Run the browser in headless mode",
        default=False,
    )

    args = parser.parse_args()

    if args.url and args.query:
        url = args.url
        query = args.query
    else:
        # Fallback to example usage if not provided
        examples = [
            ("https://www.google.com", "latest Grok model release date"),
            ("https://duckduckgo.com", "smolagents latest features 2026"),
            (
                "https://en.wikipedia.org/w/index.php?search=",
                "transformer architecture",
            ),
        ]
        url, query = examples[1]
        console.print(
            "[yellow]No --url and --query provided. Using example: "
            f"url='{url}', query='{query}'[/yellow]\n"
        )

    final_answer = run_site_search(url, query, headless=args.headless)
    console.rule("Final Summary", style="green")
    console.print(Panel(final_answer, title="Agent Summary", border_style="green"))
