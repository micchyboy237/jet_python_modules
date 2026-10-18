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
- Proper nested hierarchy for agent steps, LLM calls, and tool executions
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
-t, --temperature  LLM temperature (default: 0.3)
-mt, --max-tokens  Max tokens for LLM response (default: 500)
-tp, --top-p       Top-p sampling parameter (default: 0.95)
-pp, --presence-penalty  Presence penalty (default: 1.5)
-et, --enable-thinking   Enable thinking mode (default: False)
-ll, --log-level   Log level: DEBUG, INFO, WARNING, ERROR (default: INFO)
-tn, --trace-name  Trace/project name for telemetry (default: web-surfer-agent)
"""

import argparse
import asyncio
import json
import logging
from typing import Any, Dict

from jet.adapters.llama_cpp.config import LLM_BASE_URL, LLM_MODEL
from jet.adapters.llama_cpp.llm_utils_observed import achat
from mcp import ClientSession
from pydantic import BaseModel, Field

try:
    from mcp.client.streamable_http import streamable_http_client
except ImportError:
    from mcp.client.streamable_http import (
        streamablehttp_client as streamable_http_client,
    )
from jet_telemetry import agent, chain, get_trace_url, initialize_telemetry, tool
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace as otel_trace
from opentelemetry.trace import SpanKind

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


class AgentDecision(BaseModel):
    """Structured output model for the agent's decision."""

    thought: str = Field(description="The reasoning behind the next action")
    action: str = Field(
        description="The name of the tool to execute, or 'finish' if done"
    )
    args: Dict[str, Any] = Field(
        default_factory=dict, description="Arguments for the tool"
    )


SYSTEM_PROMPT = """
You are an expert Web Surfer Agent. Your goal is to interact with web pages to fulfill user requests.
You will be given an 'Accessibility Snapshot' of the current page.

CRITICAL RULES FOR INTELLIGENCE:
1. ALWAYS analyze the 'Current Page Snapshot' provided in the user message before deciding on an action.
2. Identify the current URL and Page Title from the snapshot to understand your location.
3. Look for specific elements (search bars, buttons, links) by their [ref=ID] tags.
4. If you are unsure where an element is, use 'browser_find' to locate it by text or selector.
5. NEVER guess the state of the page. If the snapshot shows you are on a homepage, do not try to go 'back'.
6. If an action fails, read the error message AND re-examine the snapshot to understand why.

Available Tools (parameter? = optional):
{tool_docs}

Rules:
- To press Enter after typing, use browser_type with "submit": true.
- Only use the argument names listed above. Do not invent arguments.
- If you need to see the current state of the page, use 'browser_snapshot'.
- If the snapshot is empty or incomplete, use 'browser_wait_for' with time=2 to let the page load.

Respond ONLY with a JSON object. Example:
{"thought": "I see a search bar with ref=e8. I will type my query there.", "action": "browser_type", "args": {"target": "e8", "text": "Buy Milk", "submit": true}}

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
        default=0.3,
        help="LLM temperature for response generation (default: 0.3)",
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
        default=False,
        help="Enable thinking mode in chat template (default: False)",
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


@tool(name="mcp_browser_tool", description="Execute MCP browser tool action")
async def execute_tool(session, tools_by_name, name, args):
    """
    Run one MCP browser tool with proper telemetry.
    Captures tool arguments and explicitly sets tool output for observability.
    """
    tool_obj = tools_by_name.get(name)
    if tool_obj is None:
        return False, f"Unknown tool '{name}'. Use one of: {sorted(tools_by_name)}"

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

    logger.info("🛠️ Executing tool: %s with args: %s", name, args)
    try:
        result = await session.call_tool(name, arguments=args)
        text = result.content[0].text if result.content else ""

        if span.is_recording():
            span.set_attribute("tool.output", text[:2000])
            span.set_attribute("tool.success", not is_tool_error(result, text))
            if is_tool_error(result, text):
                span.set_attribute("tool.error_message", text[:500])

        if is_tool_error(result, text):
            logger.warning("❌ Tool %s failed: %s", name, text[:100])
            return False, text

        logger.info(
            "✅ Tool %s succeeded. Output preview: %s...",
            name,
            text[:100].replace("\n", " "),
        )
        return True, text
    except Exception as e:
        if span.is_recording():
            span.set_attribute("tool.success", False)
            span.set_attribute("tool.error_message", str(e)[:500])
        logger.error("⚠️ Tool execution error for %s: %s", name, e)
        raise


async def call_llama(messages, args=None):
    """
    Send messages to local llama.cpp server using llm_utils_observed.
    Uses structured output for reliable JSON parsing.
    """
    llama_model = args.llama_model if args else LLM_MODEL
    temperature = args.temperature if args else 0.3
    max_tokens = args.max_tokens if args else 500
    top_p = args.top_p if args else 0.95
    presence_penalty = args.presence_penalty if args else 1.5
    enable_thinking = args.enable_thinking if args else False

    try:
        logger.info("🧠 Calling LLM (%s)...", llama_model)
        result = await achat(
            prompt_or_messages=messages,
            model=llama_model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            presence_penalty=presence_penalty,
            enable_thinking=enable_thinking,
            response_format=AgentDecision,
        )
        if result.structured and result.structured.success:
            decision = result.structured.parsed
            return json.dumps(decision.model_dump())
        else:
            logger.warning(
                "⚠️ Structured output failed: %s",
                result.structured.error
                if result.structured
                else "No structured result",
            )
            return result.content
    except Exception as e:
        logger.error("❌ Error calling llama.cpp via llm_utils_observed: %s", e)
        return None


@agent(name="web-surfer-loop")
async def run_agent(goal: str, args=None):
    """
    Main agent orchestration loop.
    Manages the MCP connection and delegates to session handler.
    """
    logger.info("🚀 Starting Web Surfer Agent for: %r", goal)
    span = otel_trace.get_current_span()
    if span.is_recording():
        span.set_attribute("agent.goal", goal)

    mcp_server_url = args.mcp_server_url if args else MCP_SERVER_URL
    logger.info("Connecting via Streamable HTTP to: %s", mcp_server_url)
    try:
        async with streamable_http_client(mcp_server_url) as streams:
            read_stream, write_stream = streams[0], streams[1]
            logger.info("✅ Connected to MCP Server")
            await _run_session(read_stream, write_stream, goal, args)
    except Exception as e:
        if span.is_recording():
            span.set_attribute("agent.error", str(e)[:500])
        logger.error("💥 Agent error: %s", e)
        raise


async def _wait_for_stable_snapshot(session, max_attempts=3, delay=1.0):
    """
    Wait for the page snapshot to stabilize (stop changing).
    Returns the final stable snapshot text.
    """
    last_snapshot = ""
    for i in range(max_attempts):
        result = await session.call_tool("browser_snapshot", arguments={})
        current_snapshot = result.content[0].text

        # Check if snapshot is significantly different from last one
        if current_snapshot == last_snapshot:
            logger.info("📸 Snapshot stabilized after %d attempts.", i + 1)
            return current_snapshot

        last_snapshot = current_snapshot
        if i < max_attempts - 1:
            logger.debug("⏳ Waiting %.1fs for page to stabilize...", delay)
            await asyncio.sleep(delay)

    logger.warning("⚠️ Snapshot did not fully stabilize, using latest.")
    return last_snapshot


@chain(name="web-surfer-session")
async def _run_session(read_stream, write_stream, goal: str, args=None):
    """
    Execute the web surfer session with full telemetry coverage.
    Handles initialization, navigation, and the main reasoning loop with proper nesting.
    """
    tracer = otel_trace.get_tracer(__name__)
    async with ClientSession(read_stream, write_stream) as session:
        await session.initialize()
        tools = (await session.list_tools()).tools
        tools_by_name = {t.name: t for t in tools if t.name in ALLOWED_TOOLS}
        logger.info("✅ Session initialized with %d tools", len(tools_by_name))

        session_span = otel_trace.get_current_span()
        if session_span.is_recording():
            session_span.set_attribute(
                "agent.registered_tools", json.dumps(sorted(tools_by_name.keys()))
            )
            session_span.set_attribute("agent.tools_count", len(tools_by_name))

        system_prompt = SYSTEM_PROMPT.replace(
            "{tool_docs}", build_tool_docs(tools_by_name.values())
        )

        start_url = args.start_url if args else "https://demo.playwright.dev/todomvc"
        max_steps = args.max_steps if args else MAX_STEPS
        max_repeats = args.max_repeats if args else MAX_REPEATS

        logger.info("🌐 Navigating to %s", start_url)
        await session.call_tool("browser_navigate", arguments={"url": start_url})

        # Wait for stable snapshot instead of immediate capture
        logger.info("⏳ Waiting for page to stabilize...")
        current_snapshot = await _wait_for_stable_snapshot(session)

        snap_len = len(current_snapshot)
        if snap_len < 50:
            logger.warning(
                "⚠️ Initial snapshot is very small (%d chars). Page may not have loaded.",
                snap_len,
            )
        else:
            logger.info("📸 Initial snapshot captured (%d chars).", snap_len)

        if session_span.is_recording():
            if "Page URL:" in current_snapshot:
                try:
                    url_line = [
                        line
                        for line in current_snapshot.split("\n")
                        if "Page URL:" in line
                    ][0]
                    session_span.set_attribute(
                        "browser.initial_url", url_line.split("Page URL:")[1].strip()
                    )
                except (IndexError, ValueError):
                    pass
            if "Page Title:" in current_snapshot:
                try:
                    title_line = [
                        line
                        for line in current_snapshot.split("\n")
                        if "Page Title:" in line
                    ][0]
                    session_span.set_attribute(
                        "browser.initial_title",
                        title_line.split("Page Title:")[1].strip(),
                    )
                except (IndexError, ValueError):
                    pass

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Goal: {goal}\nCurrent Page Snapshot:\n{current_snapshot}",
            },
        ]

        last_call, repeats = None, 0
        step_count = 0
        success_count = 0
        failure_count = 0

        for step in range(1, max_steps + 1):
            step_count = step
            with tracer.start_as_current_span(
                f"agent_step_{step}", kind=SpanKind.INTERNAL
            ) as step_span:
                step_span.set_attribute(
                    SpanAttributes.OPENINFERENCE_SPAN_KIND,
                    OpenInferenceSpanKindValues.AGENT.value,
                )
                step_span.set_attribute("agent.step.number", step)

                logger.info("🧠 Step %d/%d: Thinking...", step, max_steps)
                llm_response = await call_llama(messages, args)

                if not llm_response:
                    logger.error("No response from LLM, stopping")
                    step_span.set_attribute("agent.step.outcome", "llm_error")
                    break

                logger.info("🤖 LLM Decision: %s", llm_response[:100])
                try:
                    decision = json.loads(llm_response)
                except json.JSONDecodeError:
                    logger.warning("⚠️ Failed to parse JSON from LLM. Retrying...")
                    messages.append({"role": "assistant", "content": llm_response})
                    messages.append(
                        {"role": "user", "content": "Please respond with valid JSON."}
                    )
                    step_span.set_attribute("agent.step.outcome", "parse_error")
                    continue

                action = decision.get("action")
                args_dict = decision.get("args", {})
                logger.info("💡 Thought: %s", decision.get("thought"))

                if action == "finish":
                    step_span.set_attribute("agent.step.outcome", "completed")
                    step_span.set_attribute(
                        "agent.step.answer", args_dict.get("answer", "")
                    )
                    logger.info("🏁 Task Complete: %s", args_dict.get("answer"))
                    if session_span.is_recording():
                        session_span.set_attribute("agent.completion.success", True)
                        session_span.set_attribute(
                            "agent.completion.answer", args_dict.get("answer", "")
                        )
                        session_span.set_attribute("agent.steps_taken", step_count)
                        session_span.set_attribute(
                            "agent.actions_successful", success_count
                        )
                        session_span.set_attribute(
                            "agent.actions_failed", failure_count
                        )
                    break

                call_key = (action, json.dumps(args_dict, sort_keys=True))
                repeats = repeats + 1 if call_key == last_call else 1
                last_call = call_key

                if repeats >= max_repeats:
                    logger.error(
                        "🔁 Same action repeated %d times, stopping: %s",
                        repeats,
                        call_key,
                    )
                    step_span.set_attribute(
                        "agent.step.outcome", "max_repeats_exceeded"
                    )
                    if session_span.is_recording():
                        session_span.set_attribute("agent.completion.success", False)
                        session_span.set_attribute(
                            "agent.failure.reason", "max_repeats_exceeded"
                        )
                        session_span.set_attribute("agent.steps_taken", step_count)
                    break

                try:
                    ok, text = await execute_tool(
                        session, tools_by_name, action, args_dict
                    )

                    # Always refresh snapshot after action unless it was already a snapshot
                    if ok and action != "browser_snapshot":
                        logger.info("📸 Refreshing snapshot after action...")
                        current_snapshot = await _wait_for_stable_snapshot(session)
                        text = current_snapshot

                    if ok:
                        success_count += 1
                        step_span.set_attribute("agent.step.outcome", "success")
                        logger.info("✅ Action OK")
                        feedback = f"Action Result (OK):\n{text}"
                    else:
                        failure_count += 1
                        step_span.set_attribute("agent.step.outcome", "failed")
                        logger.warning("❌ Action FAILED: %s", text[:100])
                        feedback = (
                            f"Action FAILED: {text}\n"
                            f"Fix the arguments and try again.\n"
                            f"Remember to use [ref=ID] tags from the snapshot.\n"
                            f"Page is unchanged:\n{current_snapshot}"
                        )
                except Exception as e:
                    failure_count += 1
                    step_span.set_attribute("agent.step.outcome", "exception")
                    step_span.set_attribute("agent.step.error", str(e)[:500])
                    if "Session terminated" in str(e):
                        logger.error(
                            "💥 MCP session is gone (server restarted/crashed?). Stopping."
                        )
                        if session_span.is_recording():
                            session_span.set_attribute(
                                "agent.completion.success", False
                            )
                            session_span.set_attribute(
                                "agent.failure.reason", "session_terminated"
                            )
                        break
                    logger.error("⚠️ Tool execution error: %s", e)
                    ok, text = False, str(e)
                    feedback = (
                        f"Action FAILED: {text}\nPage is unchanged:\n{current_snapshot}"
                    )

                messages.append({"role": "assistant", "content": llm_response})
                messages.append({"role": "user", "content": feedback})

        else:
            logger.warning("⏹️ Reached MAX_STEPS (%d) without finishing", max_steps)
            if session_span.is_recording():
                session_span.set_attribute("agent.completion.success", False)
                session_span.set_attribute("agent.failure.reason", "max_steps_exceeded")
                session_span.set_attribute("agent.steps_taken", step_count)
                session_span.set_attribute("agent.actions_successful", success_count)
                session_span.set_attribute("agent.actions_failed", failure_count)

        logger.info("🔒 Closing MCP session gracefully...")
        try:
            await asyncio.sleep(0.5)
        except Exception:
            pass


if __name__ == "__main__":
    args = get_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Initialize telemetry with the resolved project name
    initialize_telemetry(service_name=args.trace_name)

    try:
        asyncio.run(run_agent(args.user_goal, args))
    except KeyboardInterrupt:
        logger.info("\n⚠️ Interrupted by user")
    except Exception as e:
        logger.error("💥 Fatal error: %s", e)
        raise
    finally:
        trace_url = get_trace_url()
        if trace_url:
            print(f"\n📊 View trace in Phoenix: {trace_url}")
        else:
            print("\n⚠️ No trace URL available (telemetry may not be initialized)")
