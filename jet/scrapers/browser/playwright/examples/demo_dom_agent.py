"""
Demo: LLM-driven DOM Analysis Agent
Shows how an LLM can use tools to analyze a page and recommend scraping strategies.
"""

import asyncio
from pathlib import Path

from jet.scrapers.browser.playwright.dom_agent import create_dom_agent
from rich.console import Console

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / "dom_agent_demo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


async def demo_agent_scraping_advice():
    console.rule("[bold magenta]Demo: LLM Agent Scraping Advisor[/bold magenta]")

    # Create the agent
    agent = await create_dom_agent()

    # Define the goal
    url = "https://quotes.toscrape.com/"
    goal = "I want to scrape all the quote texts and their authors from this page."

    console.print(f"[bold]Target URL:[/bold] {url}")
    console.print(f"[bold]User Goal:[/bold] {goal}\n")

    console.print("[dim]🤖 Agent is thinking and analyzing the DOM...[/dim]")

    # Let the agent reason and use tools
    advice = await agent.reason_about_page(
        source=url,
        user_goal=goal,
    )

    console.print("\n[bold green]📝 Agent's Recommended Strategy:[/bold green]")
    console.print(advice)

    # Save the advice
    output_file = OUTPUT_DIR / "scraping_advice.md"
    output_file.write_text(advice, encoding="utf-8")
    console.print(f"\n[dimm]Advice saved to: {output_file}[/dimm]")


async def demo_agent_complex_layout():
    console.rule("[bold magenta]Demo: Complex Layout Analysis[/bold magenta]")

    agent = await create_dom_agent()

    # A more complex goal requiring density analysis
    url = "https://playwright.dev/python/"
    goal = "Find the main documentation content area and ignore the navigation sidebar."

    console.print(f"[bold]Target URL:[/bold] {url}")
    console.print(f"[bold]User Goal:[/bold] {goal}\n")

    console.print("[dim]🤖 Agent is analyzing page density...[/dim]")

    advice = await agent.reason_about_page(
        source=url,
        user_goal=goal,
    )

    console.print("\n[bold green]📝 Agent's Structural Insight:[/bold green]")
    console.print(advice)


async def main():
    await demo_agent_scraping_advice()
    console.print("\n")
    await demo_agent_complex_layout()


if __name__ == "__main__":
    asyncio.run(main())
