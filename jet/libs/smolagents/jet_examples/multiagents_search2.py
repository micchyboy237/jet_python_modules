#!/usr/bin/env python
# multiagents_local_search.py
"""
Minimal multi-tool web research agent using local LLM (via llama.cpp server)

Uses:
- LiteLLM → unified interface to local OpenAI-compatible servers
- single CodeAgent (most reliable with average local models)
- rich for beautiful console output

Recommended local backends (better tool calling):
- Ollama
- LM Studio
- vLLM
- TabbyAPI

Usage:
    python multiagents_local_search.py
"""

import sys
from typing import Any

from rich.console import Console
from rich.panel import Panel

from litellm import completion
from smolagents import (
    CodeAgent,
    Model,
    WebSearchTool,
    tool,
)


console = Console()


# ──────────────────────────────────────────────────────────────────────────────
# Custom Model Wrapper for local llama.cpp / ollama / lmstudio etc.
# ──────────────────────────────────────────────────────────────────────────────

class LocalOpenAIModel(Model):
    """Adapter for local OpenAI-compatible inference servers (llama.cpp, vLLM, Ollama, LM Studio, etc.)"""

    def __init__(
        self,
        model_name: str = "local-model",
        api_base: str = "http://shawn-pc.local:8080/v1",
        api_key: str = "sk-no-key-required",
        temperature: float = 0.65,
        max_tokens: int = 4096,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.api_base = api_base.rstrip("/")  # normalize
        self.api_key = api_key
        self.default_kwargs = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }

    def __call__(self, messages: list[dict], tools: list | None = None, **kwargs: Any) -> Any:
        extra_body = {}
        if tools:
            extra_body["tools"] = tools
            extra_body["tool_choice"] = "auto"  # try "required" if model supports strongly

        call_kwargs = {
            **self.default_kwargs,
            **kwargs,
            "model": self.model_name,
            "messages": messages,
            "api_base": self.api_base,
            "api_key": self.api_key,
            "extra_body": extra_body if extra_body else None,
        }

        try:
            response = completion(**call_kwargs)
            return response
        except Exception as e:
            console.print(f"[bold red]Local model call failed:[/bold red] {e}", style="red")
            raise


# ──────────────────────────────────────────────────────────────────────────────
# Tools
# ──────────────────────────────────────────────────────────────────────────────

@tool
def visit_webpage(url: str, max_length: int = 32000) -> str:
    """Fetch a webpage and return its content as clean markdown.

    Args:
        url: The full URL of the webpage to retrieve
        max_length: Maximum length of returned markdown content (characters).
                    Longer content will be truncated. Defaults to 32000.

    Returns:
        Markdown representation of the page content, or error message
    """
    import re
    import requests
    from markdownify import markdownify

    try:
        resp = requests.get(url, timeout=12)
        resp.raise_for_status()

        md = markdownify(resp.text).strip()
        md = re.sub(r"\n{4,}", "\n\n", md)  # collapse excessive breaks

        if len(md) > max_length:
            return md[:max_length] + "\n\n… (content truncated)"

        return md

    except Exception as e:
        return f"Could not fetch page: {str(e)}"


# ──────────────────────────────────────────────────────────────────────────────
# Main Agent Setup
# ──────────────────────────────────────────────────────────────────────────────

def create_research_agent() -> CodeAgent:
    model = LocalOpenAIModel(
        model_name="whatever-you-want",  # llama.cpp usually ignores this
        api_base="http://shawn-pc.local:8080/v1",  # ← CHANGE THIS to your server
        api_key="sk-no-key-required",
        temperature=0.7,
        max_tokens=4096,
    )

    agent = CodeAgent(
        tools=[
            WebSearchTool(),
            visit_webpage,
        ],
        model=model,
        additional_authorized_imports=[
            "time",
            "datetime",
            "json",
            "re",
            "numpy",
            "pandas",
        ],
        max_steps=15,  # more generous than default
    )

    return agent


def main() -> None:
    agent = create_research_agent()

    question = (
        "If LLM training continues to scale up at the current rhythm until 2030, "
        "what would be the electric power in GW required to power the biggest "
        "training runs by 2030? What would that correspond to, compared to some "
        "countries? Please provide sources for the numbers you use."
    )

    console.print(Panel.fit(
        "[bold cyan]Research Question[/bold cyan]",
        title="Starting local agent research",
        border_style="bright_blue"
    ))
    console.print(question, style="white")
    console.print("\n[italic dim]Running agent... (this may take several minutes)[/italic dim]\n")

    try:
        with console.status("[bold green]Agent is thinking...", spinner="dots"):
            answer = agent.run(question)

        console.print("\n[bold green]━ Final Answer ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold green]")
        console.print(answer, markup=False)
        console.print("[bold green]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold green]\n")

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Interrupted by user[/bold yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[bold red]Error during execution:[/bold red]\n{str(e)}", style="red")
        sys.exit(1)


if __name__ == "__main__":
    main()