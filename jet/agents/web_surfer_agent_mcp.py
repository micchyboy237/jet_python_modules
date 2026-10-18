"""
Web Surfer MCP Agent - Automated Web Interaction with LLM

This agent uses an LLM to autonomously navigate and interact with web pages
via the Model Context Protocol (MCP) browser tools. It can perform actions
like clicking, typing, navigating, and extracting information from web pages.

The agent follows a thought-action loop where it:
    1. Receives an accessibility snapshot of the current page
    2. Uses an LLM to decide the next action
    3. Executes the action via MCP browser tools
    4. Repeats until the goal is achieved or limits are reached

Features:
    - Configurable starting URL, MCP server, and LLM settings
    - Automatic retry and error handling
    - Protection against infinite loops (max steps and repeat detection)
    - Rich console output with color-coded messages
    - OpenTelemetry tracing via jet-telemetry for observability
    - Comprehensive span coverage for all agent operations

Usage Examples:
    # Basic usage with default settings
    python web_surfer_agent_mcp.py "Add a todo item 'Buy Milk' and mark it done"

    # Custom starting URL
    python web_surfer_agent_mcp.py -u https://example.com "Find the contact page"

    # Custom MCP server and LLM model
    python web_surfer_agent_mcp.py -ms http://remote:8931/mcp -lm llama-3.2 "Search for Python docs"

    # Adjust LLM parameters for more creative responses
    python web_surfer_agent_mcp.py -t 0.7 -mt 1000 -pp 0.5 "Write a blog post draft"

    # Increase max steps for complex tasks
    python web_surfer_agent_mcp.py -s 30 -r 5 "Complete a multi-step form"

    # Minimal output (errors only)
    python web_surfer_agent_mcp.py -ll ERROR "Quick task"

    # Show all available options
    python web_surfer_agent_mcp.py --help

Command Line Arguments:
    user_goal          The task description for the agent (positional)
    -u, --url          Starting URL (default: https://demo.playwright.dev/todomvc)
    -ms, --mcp-server  MCP server URL (default: http://localhost:8931/mcp)
    -la, --llama-api   LLaMA API endpoint URL
    -lm, --llama-model LLM model name
    -s, --max-steps    Maximum iteration steps (default: 15)
    -r, --max-repeats  Max repeated actions before stopping (default: 3)
    -t, --temperature  LLM temperature (default: 0.2)
    -mt, --max-tokens  Max tokens for LLM response (default: 500)
    -tp, --top-p       Top-p sampling parameter (default: 0.95)
    -pp, --presence-penalty  Presence penalty (default: 1.5)
    -et, --enable-thinking   Enable thinking mode (default: True)
    -ll, --log-level   Log level: DEBUG, INFO, WARNING, ERROR (default: INFO)
    -tn, --trace-name  Trace/project name for telemetry (default: web-surfer-agent)
"""

import argparse
import asyncio
import json
import logging

import requests
from jet.adapters.llama_cpp.config import LLM_BASE_URL, LLM_MODEL
from mcp import ClientSession

try:
    from mcp.client.streamable_http import streamable_http_client
except ImportError:
    from mcp.client.streamable_http import (
        streamablehttp_client as streamable_http_client,
    )

# Initialize telemetry early
from jet_telemetry import get_trace_url, initialize_telemetry
from opentelemetry import trace as otel_trace

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("web_surfer_agent")
logging.getLogger("mcp").setLevel(logging.WARNING)

MCP_SERVER_URL = "http://localhost:8931/mcp"
LLAMA_API_URL = f"{LLM_BASE_URL}/chat/completions"
MAX_STEPS = 15
MAX_REPEATS = 3
ALLOWED_TOOLS = {
    "browser_navigate",
    "browser_navigate_back",
    "browser_type",
    "browser_press_key",
    "browser_hover",
    "browser_find",
    "browser_click",
    "browser_fill_form",
    "browser_select_option",
    "browser_handle_dialog",
    "browser_evaluate",
    "browser_run_code_unsafe",
    "browser_snapshot",
    "browser_take_screenshot",
    "browser_wait_for",
    "browser_close",
}
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


def get_args():
    """Parse command-line arguments with 1-2 char shorthands for keyword args."""
    parser = argparse.ArgumentParser(description="Run the Web Surfer MCP Agent.")
    parser.add_argument(
        "user_goal",
        nargs="?",
        default="Add a todo item 'Buy Milk' and then mark it as done.",
        help="Goal for the agent to accomplish (default: %(default)s)",
    )
    parser.add_argument(
        "-u",
        "--url",
        dest="start_url",
        default="https://demo.playwright.dev/todomvc",
        help="Starting URL for the browser (default: %(default)s)",
    )
    parser.add_argument(
        "-ms",
        "--mcp-server",
        dest="mcp_server_url",
        default=MCP_SERVER_URL,
        help=f"MCP server URL (default: {MCP_SERVER_URL})",
    )
    parser.add_argument(
        "-la",
        "--llama-api",
        dest="llama_api_url",
        default=LLAMA_API_URL,
        help=f"LLaMA API URL (default: {LLAMA_API_URL})",
    )
    parser.add_argument(
        "-lm",
        "--llama-model",
        dest="llama_model",
        default=LLM_MODEL,
        help=f"LLaMA model name (default: {LLM_MODEL})",
    )
    parser.add_argument(
        "-s",
        "--max-steps",
        dest="max_steps",
        type=int,
        default=MAX_STEPS,
        help=f"Maximum number of steps (default: {MAX_STEPS})",
    )
    parser.add_argument(
        "-r",
        "--max-repeats",
        dest="max_repeats",
        type=int,
        default=MAX_REPEATS,
        help=f"Maximum repeated actions before stopping (default: {MAX_REPEATS})",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        dest="temperature",
        type=float,
        default=0.2,
        help="LLM temperature for response generation (default: 0.2)",
    )
    parser.add_argument(
        "-mt",
        "--max-tokens",
        dest="max_tokens",
        type=int,
        default=500,
        help="Maximum tokens for LLM response (default: 500)",
    )
    parser.add_argument(
        "-tp",
        "--top-p",
        dest="top_p",
        type=float,
        default=0.95,
        help="Top-p sampling parameter (default: 0.95)",
    )
    parser.add_argument(
        "-pp",
        "--presence-penalty",
        dest="presence_penalty",
        type=float,
        default=1.5,
        help="Presence penalty for LLM (default: 1.5)",
    )
    parser.add_argument(
        "-et",
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        default=True,
        help="Enable thinking mode in chat template (default: True)",
    )
    parser.add_argument(
        "-ll",
        "--log-level",
        dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "-tn",
        "--trace-name",
        dest="trace_name",
        default="web-surfer-agent",
        help="Trace/project name for telemetry (default: %(default)s)",
    )
    return parser.parse_args()


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


from jet_telemetry import performance_monitor, tool


@tool(name="mcp_browser_tool", description="Execute MCP browser tool action")
@performance_monitor(threshold_ms=5000)
async def execute_tool(session, tools_by_name, name, args):
    """
    Run one MCP browser tool with proper telemetry.
    Returns (ok, text). Never raises for tool-level errors.
    """
    tool_obj = tools_by_name.get(name)
    if tool_obj is None:
        return False, f"Unknown tool '{name}'. Use one of: {sorted(tools_by_name)}"

    # Add tool-specific attributes to current span
    span = otel_trace.get_current_span()
    if span.is_recording():
        span.set_attribute("mcp.tool.name", name)
        span.set_attribute("mcp.tool.arguments", json.dumps(args, default=str))

    allowed = allowed_params(tool_obj)
    dropped = set(args) - allowed
    if dropped:
        logger.warning("✂️ Dropping unsupported args %s for %s", sorted(dropped), name)
        args = {k: v for k, v in args.items() if k in allowed}

    missing = [k for k in tool_obj.input_schema.get("required", []) if k not in args]
    if missing:
        return False, f"Missing required args {missing} for {name}."

    try:
        result = await session.call_tool(name, arguments=args)
        text = result.content[0].text if result.content else ""

        if span.is_recording():
            span.set_attribute("mcp.tool.success", not is_tool_error(result, text))
            if is_tool_error(result, text):
                span.set_attribute("mcp.tool.error_message", text[:500])

        if is_tool_error(result, text):
            return False, text
        return True, text
    except Exception as e:
        if span.is_recording():
            span.set_attribute("mcp.tool.success", False)
            span.set_attribute("mcp.tool.error_message", str(e)[:500])
        raise


@tool(
    name="mcp_session_initialize",
    description="Initialize MCP session and list available tools",
)
async def initialize_mcp_session(session):
    """Initialize MCP session and retrieve available tools with telemetry."""
    span = otel_trace.get_current_span()
    if span.is_recording():
        span.set_attribute("mcp.session.initialized", True)

    await session.initialize()
    tools = (await session.list_tools()).tools
    tools_by_name = {t.name: t for t in tools if t.name in ALLOWED_TOOLS}

    if span.is_recording():
        span.set_attribute("mcp.tools.available_count", len(tools_by_name))
        span.set_attribute("mcp.tools.names", json.dumps(sorted(tools_by_name.keys())))

    return tools_by_name


@tool(name="mcp_browser_navigate", description="Navigate browser to specified URL")
async def navigate_to_url(session, url: str):
    """Navigate to URL with telemetry tracking."""
    span = otel_trace.get_current_span()
    if span.is_recording():
        span.set_attribute("browser.url", url)
        span.set_attribute("browser.navigation.type", "navigate")

    await session.call_tool("browser_navigate", arguments={"url": url})

    if span.is_recording():
        span.set_attribute("browser.navigation.success", True)


@tool(
    name="mcp_browser_snapshot",
    description="Capture accessibility snapshot of current page",
)
async def capture_snapshot(session):
    """Capture page snapshot with telemetry."""
    result = await session.call_tool("browser_snapshot", arguments={})
    snapshot_text = result.content[0].text if result.content else ""

    span = otel_trace.get_current_span()
    if span.is_recording():
        # Extract URL and title from snapshot if present
        if "Page URL:" in snapshot_text:
            try:
                url_line = [
                    line for line in snapshot_text.split("\n") if "Page URL:" in line
                ][0]
                url = url_line.split("Page URL:")[1].strip()
                span.set_attribute("browser.snapshot.url", url)
            except (IndexError, ValueError):
                pass

        if "Page Title:" in snapshot_text:
            try:
                title_line = [
                    line for line in snapshot_text.split("\n") if "Page Title:" in line
                ][0]
                title = title_line.split("Page Title:")[1].strip()
                span.set_attribute("browser.snapshot.title", title)
            except (IndexError, ValueError):
                pass

        # Count elements in snapshot
        element_count = snapshot_text.count("[ref=")
        span.set_attribute("browser.snapshot.element_count", element_count)

    return snapshot_text


from jet_telemetry import llm


def _redact_snapshot_for_llm(snapshot: str, max_length: int = 2000) -> str:
    """Redact and truncate snapshot content for LLM messages to prevent large spans."""
    if len(snapshot) > max_length:
        return f"[SNAPSHOT TRUNCATED: {len(snapshot)} chars, showing first {max_length}]\n{snapshot[:max_length]}..."
    return snapshot


@llm(model_name="llama-cpp", provider="llama_cpp")
@performance_monitor(threshold_ms=10000)
async def call_llama(messages, args=None):
    """
    Send messages to local llama.cpp server and get response.
    Automatically redacts large snapshots from input messages.
    """
    llama_api_url = args.llama_api_url if args else LLAMA_API_URL
    llama_model = args.llama_model if args else LLM_MODEL
    temperature = args.temperature if args else 0.2
    max_tokens = args.max_tokens if args else 500
    top_p = args.top_p if args else 0.95
    presence_penalty = args.presence_penalty if args else 1.5
    enable_thinking = args.enable_thinking if args else True

    # Create redacted copy of messages for telemetry
    redacted_messages = []
    for msg in messages:
        redacted_msg = dict(msg)
        content = msg.get("content", "")
        if isinstance(content, str) and len(content) > 5000:
            # Truncate very long messages (likely containing snapshots)
            redacted_msg["content"] = _redact_snapshot_for_llm(content)
        redacted_messages.append(redacted_msg)

    payload = {
        "model": llama_model,
        "messages": redacted_messages,  # Use redacted version for telemetry
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": enable_thinking,
            },
        },
    }

    logger.info("Calling llama.cpp (%d messages)", len(messages))

    span = otel_trace.get_current_span()
    if span.is_recording():
        span.set_attribute("llm.request.message_count", len(messages))
        span.set_attribute("llm.request.model", llama_model)
        span.set_attribute("llm.request.temperature", temperature)
        span.set_attribute("llm.request.max_tokens", max_tokens)

    try:
        response = requests.post(llama_api_url, json=payload)
        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"]

        if span.is_recording():
            span.set_attribute("llm.response.success", True)
            span.set_attribute("llm.response.length_chars", len(result))

        return result
    except Exception as e:
        if span.is_recording():
            span.set_attribute("llm.response.success", False)
            span.set_attribute("llm.response.error", str(e)[:500])
        logger.error("❌ Error calling llama.cpp: %s", e)
        return None


from jet_telemetry import agent, chain


@agent(name="web-surfer-loop")
async def run_agent(goal: str, args=None):
    """
    Main agent orchestration loop.
    Manages the MCP connection and delegates to session handler.
    """
    logger.info("🚀 Starting Web Surfer Agent for: %r", goal)

    # Set agent-level attributes
    span = otel_trace.get_current_span()
    if span.is_recording():
        span.set_attribute("agent.goal", goal)
        span.set_attribute("agent.start_time", str(asyncio.get_event_loop().time()))

    mcp_server_url = args.mcp_server_url if args else MCP_SERVER_URL
    logger.info("Connecting via Streamable HTTP to: %s", mcp_server_url)

    try:
        async with streamable_http_client(mcp_server_url) as streams:
            read_stream, write_stream = streams[0], streams[1]
            logger.info("✅ Connected")
            await _run_session(read_stream, write_stream, goal, args)
    except Exception as e:
        if span.is_recording():
            span.set_attribute("agent.error", str(e)[:500])
        logger.error("💥 Agent error: %s", e)
        raise
    finally:
        if span.is_recording():
            span.set_attribute("agent.end_time", str(asyncio.get_event_loop().time()))


@chain(name="web-surfer-session")
async def _run_session(read_stream, write_stream, goal: str, args=None):
    """
    Execute the web surfer session with full telemetry coverage.
    Handles initialization, navigation, and the main reasoning loop.
    """
    async with ClientSession(read_stream, write_stream) as session:
        # Initialize session with telemetry
        logger.info("Initializing MCP session...")
        tools_by_name = await initialize_mcp_session(session)
        logger.info("✅ Session initialized with %d tools", len(tools_by_name))

        system_prompt = SYSTEM_PROMPT.replace(
            "{tool_docs}", build_tool_docs(tools_by_name.values())
        )

        start_url = args.start_url if args else "https://demo.playwright.dev/todomvc"
        max_steps = args.max_steps if args else MAX_STEPS
        max_repeats = args.max_repeats if args else MAX_REPEATS

        # Navigate to start URL with telemetry
        logger.info("🌐 Navigating to %s", start_url)
        await navigate_to_url(session, start_url)

        # Capture initial snapshot
        current_snapshot = await capture_snapshot(session)

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Goal: {goal}\n\nCurrent Page Snapshot:\n{current_snapshot}",
            },
        ]

        last_call, repeats = None, 0
        step_count = 0
        success_count = 0
        failure_count = 0

        for step in range(1, max_steps + 1):
            step_count = step
            logger.info("🧠 Step %d/%d: thinking...", step, max_steps)

            llm_response = await call_llama(messages, args)
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
            args_dict = decision.get("args", {})
            logger.info("💡 Thought: %s", decision.get("thought"))

            if action == "finish":
                logger.info("🏁 Task Complete: %s", args_dict.get("answer"))

                # Set completion attributes on chain span
                chain_span = otel_trace.get_current_span()
                if chain_span.is_recording():
                    chain_span.set_attribute(
                        "agent.completion.answer", args_dict.get("answer", "")
                    )
                    chain_span.set_attribute("agent.completion.success", True)
                    chain_span.set_attribute("agent.steps_taken", step_count)
                    chain_span.set_attribute("agent.actions_successful", success_count)
                    chain_span.set_attribute("agent.actions_failed", failure_count)

                break

            call_key = (action, json.dumps(args_dict, sort_keys=True))
            repeats = repeats + 1 if call_key == last_call else 1
            last_call = call_key

            if repeats >= max_repeats:
                logger.error(
                    "🔁 Same action repeated %d times, stopping: %s", repeats, call_key
                )

                # Set failure attributes
                chain_span = otel_trace.get_current_span()
                if chain_span.is_recording():
                    chain_span.set_attribute("agent.completion.success", False)
                    chain_span.set_attribute(
                        "agent.failure.reason", "max_repeats_exceeded"
                    )
                    chain_span.set_attribute("agent.steps_taken", step_count)

                break

            logger.info("🛠️ Executing: %s with %s", action, args_dict)

            try:
                ok, text = await execute_tool(session, tools_by_name, action, args_dict)

                if ok and action != "browser_snapshot":
                    snap = await capture_snapshot(session)
                    text = snap

                if ok:
                    success_count += 1
                    logger.info("✅ Action OK")
                    current_snapshot = text
                    feedback = f"Action Result (OK):\n{text}"
                else:
                    failure_count += 1
                    logger.warning("❌ Action FAILED: %s", text)
                    feedback = (
                        f"Action FAILED: {text}\n"
                        f"Fix the arguments and try again.\n\nPage is unchanged:\n{current_snapshot}"
                    )

            except Exception as e:
                failure_count += 1
                if "Session terminated" in str(e):
                    logger.error(
                        "💥 MCP session is gone (server restarted/crashed?). Stopping."
                    )

                    chain_span = otel_trace.get_current_span()
                    if chain_span.is_recording():
                        chain_span.set_attribute("agent.completion.success", False)
                        chain_span.set_attribute(
                            "agent.failure.reason", "session_terminated"
                        )

                    break

                logger.error("⚠️ Tool execution error: %s", e)
                ok, text = False, str(e)
                feedback = (
                    f"Action FAILED: {text}\n\nPage is unchanged:\n{current_snapshot}"
                )

            messages.append({"role": "assistant", "content": llm_response})
            messages.append({"role": "user", "content": feedback})

        else:
            logger.warning("⏹️ Reached MAX_STEPS (%d) without finishing", max_steps)

            # Set max steps reached attributes
            chain_span = otel_trace.get_current_span()
            if chain_span.is_recording():
                chain_span.set_attribute("agent.completion.success", False)
                chain_span.set_attribute("agent.failure.reason", "max_steps_exceeded")
                chain_span.set_attribute("agent.steps_taken", step_count)
                chain_span.set_attribute("agent.actions_successful", success_count)
                chain_span.set_attribute("agent.actions_failed", failure_count)

        # Handle graceful shutdown to prevent ClosedResourceError
        logger.info("🔒 Closing MCP session gracefully...")
        try:
            # Give streams time to flush any pending messages
            await asyncio.sleep(0.5)
        except Exception:
            pass  # Ignore any errors during cleanup


if __name__ == "__main__":
    args = get_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Initialize telemetry with custom trace name
    initialize_telemetry(service_name=args.trace_name)

    try:
        asyncio.run(run_agent(args.user_goal, args))
    except KeyboardInterrupt:
        logger.info("\n⚠️ Interrupted by user")
    except Exception as e:
        logger.error("💥 Fatal error: %s", e)
        raise
    finally:
        # Print trace URL for debugging/sharing
        trace_url = get_trace_url()
        if trace_url:
            print(f"\n📊 View trace in Phoenix: {trace_url}")
        else:
            print("\n⚠️ No trace URL available (telemetry may not be initialized)")
