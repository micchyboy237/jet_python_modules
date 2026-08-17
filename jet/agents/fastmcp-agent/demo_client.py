"""
Demo FastMCP client for llama_mcp_server.py

Connects to the local MCP server over stdio, discovers the tools it registers
(ask_local_llm, check_llama_server_health), lets you pick one, and executes it.

Prerequisites:
    pip install -r requirements.txt

Usage:
    python demo_client.py
    python demo_client.py --tool check_llama_server_health
    python demo_client.py --tool ask_local_llm --prompt "Explain recursion in one sentence"
"""

import argparse
import asyncio
import logging
import os

from dotenv import load_dotenv
from fastmcp import Client

load_dotenv()

# ---------------------------------------------------------------------------
# Logging — every step of discovery + execution is logged for traceability
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("demo_client")

# Path to the server script this client will spawn over stdio.
SERVER_SCRIPT = os.environ.get(
    "LLAMA_MCP_SERVER_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "llama_mcp_server.py"),
)


async def list_tools(client: Client) -> list:
    """Fetch and log the tools the connected server exposes."""
    tools = await client.list_tools()
    logger.info("Discovered %d tool(s) on server", len(tools))
    for i, tool in enumerate(tools, start=1):
        logger.info(
            "  [%d] %s — %s", i, tool.name, (tool.description or "").splitlines()[0]
        )
    return tools


def choose_tool_interactively(tools: list):
    """Prompt the user to pick a tool from the discovered list."""
    print("\nAvailable tools:")
    for i, tool in enumerate(tools, start=1):
        print(f"  {i}. {tool.name} — {(tool.description or '').splitlines()[0]}")

    while True:
        choice = input(f"\nSelect a tool (1-{len(tools)}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(tools):
            return tools[int(choice) - 1]
        print("Invalid choice, try again.")


def build_arguments_interactively(tool) -> dict:
    """Ask the user for each parameter the selected tool expects."""
    schema = tool.inputSchema or {}
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    if not properties:
        return {}

    args = {}
    print(f"\nEnter arguments for '{tool.name}' (blank = skip optional field):")
    for name, spec in properties.items():
        label = f"{name}" + (" (required)" if name in required else " (optional)")
        default = spec.get("default")
        prompt_text = f"  {label}"
        if default is not None:
            prompt_text += f" [default: {default}]"
        value = input(prompt_text + ": ").strip()

        if value == "" and name not in required:
            continue
        # Cast to the declared JSON schema type where reasonable.
        param_type = spec.get("type")
        if param_type == "integer" and value:
            value = int(value)
        args[name] = value
    return args


async def run_tool(client: Client, tool_name: str, arguments: dict):
    logger.info("Calling tool '%s' with arguments=%s", tool_name, arguments)
    result = await client.call_tool(tool_name, arguments)
    logger.info("Tool '%s' returned successfully", tool_name)
    return result


async def main():
    parser = argparse.ArgumentParser(
        description="Demo MCP client for the local llama.cpp bridge"
    )
    parser.add_argument(
        "--tool", help="Tool name to run non-interactively (e.g. ask_local_llm)"
    )
    parser.add_argument(
        "--prompt", help="Prompt text, used when --tool ask_local_llm is set"
    )
    args = parser.parse_args()

    logger.info("Connecting to MCP server via stdio: %s", SERVER_SCRIPT)
    client = Client(SERVER_SCRIPT)

    async with client:
        tools = await list_tools(client)
        if not tools:
            logger.error("Server exposed no tools — is llama_mcp_server.py correct?")
            return

        if args.tool:
            # Non-interactive path, e.g. for scripting/CI.
            tool = next((t for t in tools if t.name == args.tool), None)
            if tool is None:
                logger.error("No tool named '%s' on this server", args.tool)
                return
            call_args = {}
            if args.tool == "ask_local_llm":
                if not args.prompt:
                    logger.error("--prompt is required when using --tool ask_local_llm")
                    return
                call_args = {"prompt": args.prompt}
        else:
            # Interactive path: choose tool, then fill in its arguments.
            tool = choose_tool_interactively(tools)
            call_args = build_arguments_interactively(tool)

        result = await run_tool(client, tool.name, call_args)

        print("\n--- Result ---")
        for block in result.content:
            if hasattr(block, "text"):
                print(block.text)
            else:
                print(block)


if __name__ == "__main__":
    asyncio.run(main())
