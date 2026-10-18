import asyncio
import json
import logging

from mcp import ClientSession

try:
    from mcp.client.streamable_http import streamable_http_client
except ImportError:
    from mcp.client.streamable_http import (
        streamablehttp_client as streamable_http_client,
    )
from jet.adapters.llama_cpp.llm_utils_observed import achat
from jet_telemetry import agent, chain
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace as otel_trace
from opentelemetry.trace import SpanKind

from .config import (
    ALLOWED_TOOLS,
    DEFAULT_LLM_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PRESENCE_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
)
from .tools import build_tool_docs, execute_tool
from .types import SYSTEM_PROMPT_TEMPLATE, AgentDecision

logger = logging.getLogger("web_surfer_agent")


async def call_llama(messages, args=None):
    """
    Send messages to local llama.cpp server using llm_utils_observed.
    Uses structured output for reliable JSON parsing.
    """

    llama_model = args.llama_model if args else DEFAULT_LLM_MODEL
    temperature = args.temperature if args else DEFAULT_TEMPERATURE
    max_tokens = args.max_tokens if args else DEFAULT_MAX_TOKENS
    top_p = args.top_p if args else DEFAULT_TOP_P
    presence_penalty = args.presence_penalty if args else DEFAULT_PRESENCE_PENALTY
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

    from .config import DEFAULT_MCP_SERVER_URL

    mcp_server_url = args.mcp_server_url if args else DEFAULT_MCP_SERVER_URL
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

        system_prompt = SYSTEM_PROMPT_TEMPLATE.replace(
            "{tool_docs}", build_tool_docs(tools_by_name.values())
        )

        from .config import DEFAULT_MAX_REPEATS, DEFAULT_MAX_STEPS, DEFAULT_START_URL

        start_url = args.start_url if args else DEFAULT_START_URL
        max_steps = args.max_steps if args else DEFAULT_MAX_STEPS
        max_repeats = args.max_repeats if args else DEFAULT_MAX_REPEATS

        logger.info("🌐 Navigating to %s", start_url)
        await session.call_tool("browser_navigate", arguments={"url": start_url})

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
