"""
Web Surfer Agent with MCP Support

Usage Examples:

1. Find the top story on Hacker News:
   python -m jet.agents.web_surfer_agent_mcp.main "Find the top story on Hacker News and tell me its title."

2. Search for a specific topic on Google:
   python -m jet.agents.web_surfer_agent_mcp.main "Search for 'OpenTelemetry best practices' on Google and summarize the first result."

3. Fill out a form (hypothetical):
   python -m jet.agents.web_surfer_agent_mcp.main "Go to example.com/form, fill 'Name' with 'John', 'Email' with 'john@example.com', and submit."
"""

import argparse
import asyncio
import sys

# Set up sys.path for external microsoft-agent-framework modules (added as per file context)
sys.path.insert(
    0,
    "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/microsoft-agent-framework/python/packages/core",
)
sys.path.insert(
    0,
    "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/microsoft-agent-framework/python/packages/openai",
)
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from jet.adapters.llama_cpp.config import LLM_BASE_URL, LLM_MODEL

from .tools import get_mcp_browser_tool


def get_args():
    parser = argparse.ArgumentParser(
        description="Web Surfer Agent: Autonomous browser agent using MCP."
    )
    parser.add_argument(
        "goal",
        type=str,
        help="The goal for the agent to achieve (e.g., 'Find the top story on Hacker News').",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=LLM_MODEL,
        help=f"LLM model to use (default: {LLM_MODEL}).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=15,
        help="Maximum number of steps/iterations for the agent.",
    )
    return parser.parse_args()


async def main():
    args = get_args()

    # Get the MCP browser tool
    browser_tool = get_mcp_browser_tool()

    # Create the agent
    # Note: You need to configure your ChatClient appropriately for your LLM provider
    # For llama.cpp via OpenAI-compatible API, you might use a generic OpenAIChatClient
    # pointing to your local server.
    client = OpenAIChatClient(
        base_url=LLM_BASE_URL,  # Example llama.cpp endpoint
        api_key="not-needed",
        model=args.model,
    )

    async with Agent(
        client=client,
        name="WebSurfer",
        instructions=(
            "You are a helpful web surfer agent. Use the provided browser tools to achieve the user's goal. "
            "Always start by navigating to the relevant website. "
            "Use browser_snapshot to understand the page state before interacting. "
            "Be precise with your actions."
        ),
        tools=browser_tool,
    ) as agent:
        print(f"🤖 Agent starting with goal: {args.goal}")
        response = await agent.run(args.goal)
        print(f"\n✅ Final Result:\n{response.text}")


if __name__ == "__main__":
    asyncio.run(main())
