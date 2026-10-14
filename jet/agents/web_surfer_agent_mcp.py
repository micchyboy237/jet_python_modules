# jet_python_modules/jet/agents/web_surfer_agent_mcp.py
import asyncio
import json
import logging

import requests
from jet.adapters.llama_cpp.config import LLM_BASE_URL, LLM_MODEL
from mcp import ClientSession

try:
    from mcp.client.streamable_http import streamable_http_client  # newer SDK
except ImportError:
    from mcp.client.streamable_http import (  # older SDK name
        streamablehttp_client as streamable_http_client,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("web_surfer_agent")
logging.getLogger("mcp").setLevel(
    logging.WARNING
)  # hides the huge tools/list and message dumps

MCP_SERVER_URL = "http://localhost:8931/mcp"
LLAMA_API_URL = f"{LLM_BASE_URL}/chat/completions"

MAX_STEPS = 15
MAX_REPEATS = 3  # same action N times in a row => stop
ALLOWED_TOOLS = {
    "browser_navigate",
    "browser_click",
    "browser_type",
    "browser_press_key",
    "browser_snapshot",
}
# Optional args the LLM may use. Everything else optional is hidden and stripped.
SAFE_OPTIONAL = {"browser_type": {"submit"}}

SYSTEM_PROMPT = """
You are a Web Surfer Agent. Your goal is to interact with web pages to fulfill user requests.
You will be given an 'Accessibility Snapshot' of the current page.
Elements have reference IDs like [ref=e5]. Pass that ID (e.g. "e5") as the "target" argument.

Available Tools (parameter? = optional):
{tool_docs}

Rules:
- To press Enter after typing, use browser_type with "submit": true.
- Only use the argument names listed above. Do not invent arguments.

Respond ONLY with a JSON object. Example:
{"thought": "Type the todo and press Enter", "action": "browser_type", "args": {"target": "e8", "text": "Buy Milk", "submit": true}}

If an action FAILED, read the error and fix your arguments.
When the task is complete, use: {"thought": "...", "action": "finish", "args": {"answer": "..."}}
"""


def allowed_params(tool) -> set:
    """Required args + the few optional args we trust the LLM with."""
    required = set(tool.input_schema.get("required", []))
    return required | SAFE_OPTIONAL.get(tool.name, set())


def build_tool_docs(tools) -> str:
    """Build the tool list for the prompt from the server's own schemas."""
    lines = []
    for t in tools:
        props = t.input_schema.get("properties", {})
        required = set(t.input_schema.get("required", []))
        params = ", ".join(
            f"{name}{'' if name in required else '?'}: {props[name].get('type', 'any')}"
            for name in props
            if name in allowed_params(t)
        )
        lines.append(f"- {t.name}({params}): {t.description}")
    return "\n".join(lines)


def is_tool_error(result, text: str) -> bool:
    """Works across SDK versions (is_error vs isError) and falls back to the text."""
    flag = getattr(result, "is_error", None)
    if flag is None:
        flag = getattr(result, "isError", None)
    return bool(flag) or text.lstrip().startswith("### Error")


async def execute_tool(session, tools_by_name, name, args):
    """Run one tool. Returns (ok, text). Never raises for tool-level errors."""
    tool = tools_by_name.get(name)
    if tool is None:
        return False, f"Unknown tool '{name}'. Use one of: {sorted(tools_by_name)}"

    allowed = allowed_params(tool)
    dropped = set(args) - allowed
    if dropped:
        logger.warning("✂️ Dropping unsupported args %s for %s", sorted(dropped), name)
        args = {k: v for k, v in args.items() if k in allowed}

    missing = [k for k in tool.input_schema.get("required", []) if k not in args]
    if missing:
        return False, f"Missing required args {missing} for {name}."

    result = await session.call_tool(name, arguments=args)
    text = result.content[0].text if result.content else ""
    if is_tool_error(result, text):
        return False, text
    return True, text


async def call_llama(messages):
    """Send messages to local llama.cpp server and get response."""
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 500,
        "top_p": 0.95,
        "presence_penalty": 1.5,
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": True,
            },
        },
    }
    logger.info("Calling llama.cpp (%d messages)", len(messages))
    try:
        response = requests.post(LLAMA_API_URL, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error("❌ Error calling llama.cpp: %s", e)
        return None


async def run_agent(goal: str):
    logger.info("🚀 Starting Web Surfer Agent for: %r", goal)
    logger.info("Connecting via Streamable HTTP to: %s", MCP_SERVER_URL)

    # `async with` opens AND closes the connection for us
    async with streamable_http_client(MCP_SERVER_URL) as streams:
        read_stream, write_stream = (
            streams[0],
            streams[1],
        )  # 3rd item (session id getter) unused
        logger.info("✅ Connected")
        await _run_session(read_stream, write_stream, goal)


async def _run_session(read_stream, write_stream, goal: str):
    async with ClientSession(read_stream, write_stream) as session:
        await session.initialize()
        logger.info("✅ Session initialized")

        # Ask the server what tools really exist, then keep only the ones we need
        tools = (await session.list_tools()).tools
        tools_by_name = {t.name: t for t in tools if t.name in ALLOWED_TOOLS}
        logger.info("🧰 Tools given to LLM: %s", sorted(tools_by_name))
        system_prompt = SYSTEM_PROMPT.replace(
            "{tool_docs}", build_tool_docs(tools_by_name.values())
        )

        start_url = "https://demo.playwright.dev/todomvc"
        logger.info("🌐 Navigating to %s", start_url)
        await session.call_tool("browser_navigate", arguments={"url": start_url})
        result = await session.call_tool("browser_snapshot", arguments={})
        current_snapshot = result.content[0].text

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Goal: {goal}\n\nCurrent Page Snapshot:\n{current_snapshot}",
            },
        ]

        last_call, repeats = None, 0
        for step in range(1, MAX_STEPS + 1):
            logger.info("🧠 Step %d/%d: thinking...", step, MAX_STEPS)
            llm_response = await call_llama(messages)
            if not llm_response:
                logger.error("No response from LLM, stopping")
                break
            logger.info("🤖 LLM Response: %s", llm_response)

            try:
                clean_json = (
                    llm_response.replace("```json", "").replace("```", "").strip()
                )
                decision = json.loads(clean_json)
            except json.JSONDecodeError:
                logger.warning("⚠️ Failed to parse JSON from LLM. Retrying...")
                messages.append({"role": "assistant", "content": llm_response})
                messages.append(
                    {"role": "user", "content": "Please respond with valid JSON."}
                )
                continue

            action = decision.get("action")
            args = decision.get("args", {})
            logger.info("💡 Thought: %s", decision.get("thought"))

            if action == "finish":
                logger.info("🏁 Task Complete: %s", args.get("answer"))
                break

            # Stop if the model is stuck repeating the exact same call
            call_key = (action, json.dumps(args, sort_keys=True))
            repeats = repeats + 1 if call_key == last_call else 1
            last_call = call_key
            if repeats >= MAX_REPEATS:
                logger.error(
                    "🔁 Same action repeated %d times, stopping: %s", repeats, call_key
                )
                break

            logger.info("🛠️ Executing: %s with %s", action, args)
            try:
                ok, text = await execute_tool(session, tools_by_name, action, args)
                if ok and action != "browser_snapshot":
                    snap = await session.call_tool("browser_snapshot", arguments={})
                    text = snap.content[0].text
            except Exception as e:
                if "Session terminated" in str(e):
                    logger.error(
                        "💥 MCP session is gone (server restarted/crashed?). Stopping."
                    )
                    break
                logger.error("⚠️ Tool execution error: %s", e)
                ok, text = False, str(e)

            if ok:
                logger.info("✅ Action OK")
                current_snapshot = text
                feedback = f"Action Result (OK):\n{text}"
            else:
                logger.warning("❌ Action FAILED: %s", text)
                feedback = (
                    f"Action FAILED: {text}\n"
                    f"Fix the arguments and try again.\n\nPage is unchanged:\n{current_snapshot}"
                )

            messages.append({"role": "assistant", "content": llm_response})
            messages.append({"role": "user", "content": feedback})
        else:
            logger.warning("⏹️ Reached MAX_STEPS (%d) without finishing", MAX_STEPS)


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Web Surfer MCP Agent.")
    parser.add_argument(
        "user_goal",
        nargs="?",
        default="Add a todo item 'Buy Milk' and then mark it as done.",
        help="Goal for the agent to accomplish (default: %(default)s)",
    )
    args = parser.parse_args()
    asyncio.run(run_agent(args.user_goal))
