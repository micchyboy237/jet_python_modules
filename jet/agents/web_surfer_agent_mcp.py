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

MCP_SERVER_URL = "http://localhost:8931/mcp"
LLAMA_API_URL = f"{LLM_BASE_URL}/chat/completions"

SYSTEM_PROMPT = """
You are a Web Surfer Agent. Your goal is to interact with web pages to fulfill user requests.
You will be provided with an 'Accessibility Snapshot' of the current page. 
Elements in the snapshot have reference IDs like [ref=e5]. Use these IDs for clicking or typing.
Available Tools:
1. browser_navigate(url): Opens a URL.
2. browser_click(ref): Clicks an element by its reference ID (e.g., 'e5').
3. browser_type(ref, text): Types text into an element by its reference ID.
4. browser_press_key(key): Presses a keyboard key (e.g., 'Enter').
5. browser_snapshot(): Takes a new accessibility snapshot of the current page.
Respond ONLY with a JSON object in this format:
{
    "thought": "Your reasoning about what to do next",
    "action": "tool_name",
    "args": { "arg_name": "value" }
}
If the task is complete, set "action" to "finish" and provide the final answer in "args".
"""


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

        start_url = "https://demo.playwright.dev/todomvc"
        logger.info("🌐 Navigating to %s", start_url)
        await session.call_tool("browser_navigate", arguments={"url": start_url})
        result = await session.call_tool("browser_snapshot", arguments={})
        current_snapshot = result.content[0].text

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Goal: {goal}\n\nCurrent Page Snapshot:\n{current_snapshot}",
            },
        ]

        step = 0
        while True:
            step += 1
            logger.info("🧠 Step %d: thinking...", step)
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
            thought = decision.get("thought")
            logger.info("💡 Thought: %s", thought)

            if action == "finish":
                logger.info("🏁 Task Complete: %s", args.get("answer"))
                break
            elif action == "browser_snapshot":
                result = await session.call_tool(action, arguments=args)
                current_snapshot = result.content[0].text
            else:
                logger.info("🛠️ Executing: %s with %s", action, args)
                try:
                    await session.call_tool(action, arguments=args)
                    snap_result = await session.call_tool(
                        "browser_snapshot", arguments={}
                    )
                    current_snapshot = snap_result.content[0].text
                except Exception as e:
                    logger.error("⚠️ Tool execution error: %s", e)
                    current_snapshot = f"Error executing {action}: {str(e)}"

            messages.append({"role": "assistant", "content": llm_response})
            messages.append(
                {"role": "user", "content": f"Action Result:\n{current_snapshot}"}
            )


if __name__ == "__main__":
    user_goal = "Add a todo item 'Buy Milk' and then mark it as done."
    asyncio.run(run_agent(user_goal))
