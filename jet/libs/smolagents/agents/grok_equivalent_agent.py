# grok_equivalent_agent.py

import logging
import re
from pathlib import Path
from typing import Literal

import requests
from markdownify import markdownify
from requests.exceptions import RequestException
from tqdm import tqdm

try:
    import fitz  # pymupdf - pip install pymupdf for PDF text extraction support
except ImportError:
    fitz = None
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


def _fetch_url(url: str):
    """Internal helper to fetch raw bytes and content-type from a URL with a realistic User-Agent."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        return response.content, content_type
    except RequestException as e:
        raise ValueError(f"Error fetching the resource: {str(e)}") from e


def _process_to_markdown(raw_content: str) -> str:
    """Internal method to convert raw HTML to cleaned Markdown."""
    markdown_content = markdownify(raw_content).strip()
    markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)
    return markdown_content


@tool
def visit_webpage(url: str) -> str:
    """
    Visits a resource at the given URL and returns its content in a readable format.

    Supported resource types:
    • HTML pages → converted to cleaned Markdown
    • PDF files   → text extracted with PyMuPDF (optional tqdm progress for large docs)
    • Other types → helpful message with the direct URL (and installation note if PDF)

    This handles common cases seen in search results (paywalled HTML, direct PDFs, binary attachments).

    Args:
        url (str): The full URL of the resource to visit (HTML page, PDF, etc.).

    Returns:
        str: The processed content (Markdown for HTML, plain text for PDF) or an error/helpful message.

    Note:
        For progress in loops (e.g., multiple URLs), use tqdm in your calling code:
        from tqdm import tqdm
        for url in tqdm(urls): ...
    """
    try:
        content_bytes, content_type = _fetch_url(url)
    except ValueError as e:
        return str(e)

    # Primary detection via Content-Type, fallback to URL extension
    is_pdf = ("application/pdf" in content_type) or url.lower().endswith(".pdf")

    if is_pdf:
        if fitz is None:
            return (
                "Detected a PDF resource, but 'pymupdf' is not installed. "
                "Run `pip install pymupdf` to enable automatic text extraction. "
                f"Direct download URL: {url}"
            )
        try:
            doc = fitz.open(stream=content_bytes, filetype="pdf")
            pages = []
            # Show tqdm progress only for reasonably large PDFs to avoid clutter
            iterator = (
                tqdm(doc, desc="Extracting PDF pages", leave=False)
                if doc.page_count > 10
                else doc
            )
            for page in iterator:
                pages.append(page.get_text("text"))
            full_text = "\n\n".join(pages)
            full_text = re.sub(r"\n{3,}", "\n\n", full_text.strip())
            return (
                f"# Extracted Text from PDF ({doc.page_count} pages)\n\n{full_text}"
                if full_text
                else "PDF processed but no extractable text found."
            )
        except Exception as e:
            return f"Error extracting text from PDF: {str(e)}. Direct URL: {url}"

    elif "text/html" in content_type:
        try:
            html_text = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            html_text = content_bytes.decode("utf-8", errors="replace")
        return _process_to_markdown(html_text)

    # Fallback for any other content type
    return (
        f"Resource fetched successfully but content type '{content_type}' is not directly convertible to text. "
        f"Direct URL for manual access: {url}"
    )


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
    # Lazy import to avoid circular import at module top-level (since create_local_model may depend on the agent)
    from jet.libs.smolagents.agents.controlled_messages_agent import create_local_model

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
    # Lazy import to avoid circular import at module top-level
    from jet.libs.smolagents.utils.model_utils import create_local_model

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
