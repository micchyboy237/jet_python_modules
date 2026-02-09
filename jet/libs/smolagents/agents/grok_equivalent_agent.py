# grok_equivalent_agent.py

import logging
import re
from pathlib import Path
from typing import Literal

import requests
from jet.libs.smolagents.agents.controlled_messages_agent import (
    create_local_model,  # For older Python; use typing in 3.12+
)
from markdownify import markdownify
from requests.exceptions import RequestException
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from smolagents import (
    CodeAgent,
    InferenceClientModel,
    ToolCallingAgent,
    WebSearchTool,
    tool,
)
from typing_extensions import TypedDict

# Set up logging with RichHandler for beautiful output
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("grok_equivalent")

console = Console()


_instructions_path = Path(__file__).parent / "instructions" / "grok_equivalent.md"
with open(_instructions_path, encoding="utf-8") as f:
    FULL_INSTRUCTIONS = f.read()


class AgentConfig(TypedDict):
    model_id: str
    max_steps: int
    instructions: str | None
    executor_type: Literal["local", "blaxel", "e2b", "modal", "docker", "wasm"]


def _fetch_url(url: str) -> str:
    """Internal method to fetch raw content from a URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except RequestException as e:
        raise ValueError(f"Error fetching the webpage: {str(e)}") from e
    except Exception as e:
        raise ValueError(f"An unexpected error occurred: {str(e)}") from e


def _process_to_markdown(raw_content: str) -> str:
    """Internal method to convert raw HTML to cleaned Markdown."""
    markdown_content = markdownify(raw_content).strip()
    markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)
    return markdown_content


@tool
def visit_webpage(url: str) -> str:
    """Visits a webpage at the given URL and returns its content as a markdown string.

    Args:
        url: The URL of the webpage to visit.

    Returns:
        The content of the webpage converted to Markdown, or an error message if the request fails.

    Note: For progress in loops (e.g., multiple URLs), use tqdm in your calling code: from tqdm import tqdm; for url in tqdm(urls): ...
    """
    try:
        raw_content = _fetch_url(url)
        return _process_to_markdown(raw_content)
    except ValueError as e:
        return str(e)


def create_web_agent(
    model: InferenceClientModel | None,
    tools: list = None,
    max_steps: int = 10,
    name: str = "web_search_agent",
    description: str = "Handles web searches and page browsing for gathering information.",
) -> ToolCallingAgent:
    """Creates a ToolCallingAgent for web-related tasks.

    To override defaults: Pass custom tools list, e.g., tools=[WebSearchTool(), my_custom_tool].
    """
    if model is None:
        model = create_local_model(agent_name=name)
    if tools is None:
        tools = [WebSearchTool(), visit_webpage]
    return ToolCallingAgent(
        tools=tools,
        model=model,
        max_steps=max_steps,
        name=name,
        description=description,
    )


def create_manager_agent(
    model: InferenceClientModel | None,
    managed_agents: list,
    instructions: str = FULL_INSTRUCTIONS,  # Now uses the full, detailed prompt by default
    additional_authorized_imports: list[str] = [
        "time",
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
        "sympy",
    ],
    executor_type: Literal[
        "local", "blaxel", "e2b", "modal", "docker", "wasm"
    ] = "local",
) -> CodeAgent:
    """Creates a CodeAgent as the manager for deep reasoning and orchestration.

    To override the full default instructions: Pass a custom string via the instructions parameter or via config['instructions'].
    Example:
        config['instructions'] = "Your custom system prompt here..."
    """
    if model is None:
        model = create_local_model(agent_name="manager_agent")
    return CodeAgent(
        tools=[],
        model=model,
        managed_agents=managed_agents,
        instructions=instructions or FULL_INSTRUCTIONS,  # Fallback to full if None
        additional_authorized_imports=additional_authorized_imports,
        executor_type=executor_type,
        add_base_tools=True,
    )


def display_config_table(config: AgentConfig) -> None:
    """Displays the config in a rich table for beautiful output."""
    table = Table(title="Agent Configuration")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="magenta")
    for key, value in config.items():
        table.add_row(key, str(value))
    console.print(table)


def main(config: AgentConfig, query: str | None = None) -> None:
    """Main function to set up and run the multi-agent system.

    To override defaults: Update config dict before calling, e.g., config['model_id'] = 'new-model-id'; config['instructions'] = 'custom instructions'.
    Pass custom query to override the default example.
    """
    logger.info("Setting up multi-agent system...")
    display_config_table(config)

    # Initialize model (requires HF token; run huggingface_hub.login() if needed)
    # model = InferenceClientModel(model_id=config["model_id"])
    model = None

    # Create web agent
    web_agent = create_web_agent(model=model, max_steps=config["max_steps"])

    # Create manager agent
    manager_agent = create_manager_agent(
        model=model,
        managed_agents=[web_agent],
        instructions=config.get("instructions"),
        executor_type=config.get("executor_type", "local"),
    )

    # Use provided query or default example
    if query is None:
        query = (
            "If LLM training continues to scale up at the current rhythm until 2030, what would be the electric power in GW "
            "required to power the biggest training runs by 2030? What would that correspond to, compared to some countries? "
            "Please provide a source for any numbers used."
        )

    logger.info(f"Running query: {query}")
    answer = manager_agent.run(query)

    # Print result with rich for beautiful output
    console.print("\n[bold blue]Final Answer:[/bold blue]")
    console.print(answer)


if __name__ == "__main__":
    config: AgentConfig = {
        "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",  # Free HF model; override via config['model_id'] = 'new-model'
        "max_steps": 20,
        "instructions": None,  # Override via config['instructions'] = "custom prompt"
        "executor_type": "local",
    }
    main(config)
