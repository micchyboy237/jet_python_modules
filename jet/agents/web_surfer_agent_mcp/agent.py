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
from jet.adapters.llama_cpp.rag_utils.query_snapshot import find_snapshot
from jet.adapters.llama_cpp.token_utils import count_tokens
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


async def _generate_search_query(
    goal: str, last_thought: str | None, current_url: str
) -> str:
    """
    Ask the LLM to generate a concise, keyword-rich search query for RAG retrieval
    from the ARIA snapshot. Focuses on UI elements, labels, roles, and states.
    """
    query_prompt = f"""
You are an expert at searching web page accessibility snapshots. 
Goal: {goal}
Current URL: {current_url}
Last Thought: {last_thought or "N/A"}

Generate a short, keyword-based search query to find relevant UI elements in the page snapshot. 
Focus on:
- Element roles (button, input, link, heading, listitem)
- Visible text labels or placeholders
- States (disabled, checked, selected)
- Structural containers (navbar, footer, modal, form)

Example:
Goal: "Log in with email test@example.com"
Query: "input email password button login sign in form"

Return ONLY the query string.
"""
    try:
        # Use a simple chat call without structured output for speed
        result = await achat(
            prompt_or_messages=[{"role": "user", "content": query_prompt}],
            model=DEFAULT_LLM_MODEL,  # Or a smaller/faster model if available
            temperature=0.1,
            max_tokens=50,
        )
        query = result.content.strip().strip('"').strip("'")
        logger.debug(f"🔍 Generated RAG Query: {query}")
        return query
    except Exception as e:
        logger.warning(f"Failed to generate search query: {e}. Falling back to goal.")
        return goal


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


async def _wait_for_stable_snapshot(session, max_attempts=8, delay=1.0):
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


def _get_rag_context(query: str, snapshot: str, top_k: int = 5) -> str:
    """
    Retrieve relevant chunks from the snapshot using vector search.
    Falls back to a truncated snapshot if no relevant chunks are found.
    """
    if not snapshot:
        return ""

    try:
        rag_context, _, _ = find_snapshot(
            query=query, text=snapshot, top_k=top_k, min_score=0.15
        )
        if rag_context:
            return rag_context

        # Fallback: Take first few lines if no relevant chunks found
        lines = snapshot.split("\n")[:50]
        return (
            "\n".join(lines)
            + "\n...(context truncated, use browser_snapshot if needed)"
        )
    except Exception as e:
        logger.warning(f"RAG context retrieval failed: {e}. Using truncated snapshot.")
        return snapshot[:2000] + "\n...(truncated due to RAG error)"


async def _initialize_session(session, goal: str, args=None):
    """
    Initialize the MCP session, register tools, navigate to start URL,
    and capture the initial stable snapshot.
    Returns:
        tuple: (tools_by_name, system_prompt, initial_messages, current_snapshot, session_span)
    """
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

    from .config import DEFAULT_START_URL

    start_url = args.start_url if args else DEFAULT_START_URL
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
                    line for line in current_snapshot.split("\n") if "Page URL:" in line
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

    # Generate an optimized search query for the initial context
    initial_query = await _generate_search_query(goal, None, start_url)
    initial_context = _get_rag_context(
        query=initial_query, snapshot=current_snapshot, top_k=5
    )

    initial_messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Goal: {goal}\nRelevant Page Context:\n{initial_context}",
        },
    ]
    return (
        tools_by_name,
        system_prompt,
        initial_messages,
        current_snapshot,
        session_span,
    )


def _trim_messages(messages: list, max_tokens: int, model: str) -> list:
    """
    Trim message history to fit within max_tokens.
    Always keeps the system prompt (index 0) and the most recent messages.
    """
    if len(messages) <= 1:
        return messages

    system_msg = messages[0]
    conversation_history = messages[1:]

    system_tokens = count_tokens(
        content=system_msg["content"], model=model, use_server=False
    )
    remaining_budget = max_tokens - system_tokens - 500

    if remaining_budget <= 0:
        logger.warning(
            "System prompt alone exceeds token budget. Truncating system prompt."
        )
        return [system_msg]

    trimmed_history = []
    current_tokens = 0
    for msg in reversed(conversation_history):
        msg_tokens = count_tokens(content=msg["content"], model=model, use_server=False)
        if current_tokens + msg_tokens <= remaining_budget:
            trimmed_history.append(msg)
            current_tokens += msg_tokens
        else:
            break

    trimmed_history.reverse()
    return [system_msg] + trimmed_history


async def _execute_step(
    step: int,
    max_steps: int,
    messages: list,
    args,
    session,
    tools_by_name: dict,
    current_snapshot: str,
    last_call: tuple | None,
    repeats: int,
    max_repeats: int,
    tracer,
    last_decision: dict | None = None,
    goal: str = "",
):
    """
    Execute a single step of the agent loop.
    Includes token management and RAG-based context injection to prevent context overflow.
    """
    step_count = step
    success_count = 0
    failure_count = 0

    with tracer.start_as_current_span(
        f"agent_step_{step}", kind=SpanKind.INTERNAL
    ) as step_span:
        step_span.set_attribute(
            SpanAttributes.OPENINFERENCE_SPAN_KIND,
            OpenInferenceSpanKindValues.AGENT.value,
        )
        step_span.set_attribute("agent.step.number", step)

        llama_model = args.llama_model if args else DEFAULT_LLM_MODEL
        max_tokens_limit = args.max_tokens if args else DEFAULT_MAX_TOKENS

        original_len = len(messages)
        managed_messages = _trim_messages(messages, max_tokens_limit, llama_model)
        if len(managed_messages) < original_len:
            dropped = original_len - len(managed_messages)
            logger.warning(
                "📉 Context trimmed: Dropped %d old messages to fit token limit.",
                dropped,
            )
            step_span.set_attribute("agent.context_trimmed", True)
            step_span.set_attribute("agent.messages_dropped", dropped)

        logger.info("🧠 Step %d/%d: Thinking...", step, max_steps)
        llm_response = await call_llama(managed_messages, args)

        if not llm_response:
            logger.error("No response from LLM, stopping")
            step_span.set_attribute("agent.step.outcome", "llm_error")
            return {
                "should_break": True,
                "break_reason": "llm_error",
                "messages": messages,
                "current_snapshot": current_snapshot,
                "last_call": last_call,
                "repeats": repeats,
                "success_delta": 0,
                "failure_delta": 0,
                "decision": None,
            }

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
            return {
                "should_break": False,
                "messages": messages,
                "current_snapshot": current_snapshot,
                "last_call": last_call,
                "repeats": repeats,
                "success_delta": 0,
                "failure_delta": 0,
                "decision": None,
            }

        action = decision.get("action")
        args_dict = decision.get("args", {})
        thought = decision.get("thought", "")

        logger.info("💡 Thought: %s", thought)

        if action == "finish":
            step_span.set_attribute("agent.step.outcome", "completed")
            step_span.set_attribute("agent.step.answer", args_dict.get("answer", ""))
            logger.info("🏁 Task Complete: %s", args_dict.get("answer"))
            return {
                "should_break": True,
                "break_reason": "completed",
                "answer": args_dict.get("answer", ""),
                "messages": messages,
                "current_snapshot": current_snapshot,
                "last_call": last_call,
                "repeats": repeats,
                "success_delta": 0,
                "failure_delta": 0,
                "decision": decision,
            }

        call_key = (action, json.dumps(args_dict, sort_keys=True))
        new_repeats = repeats + 1 if call_key == last_call else 1
        new_last_call = call_key

        if new_repeats >= max_repeats:
            logger.error(
                "🔁 Same action repeated %d times, stopping: %s",
                new_repeats,
                call_key,
            )
            step_span.set_attribute("agent.step.outcome", "max_repeats_exceeded")
            return {
                "should_break": True,
                "break_reason": "max_repeats_exceeded",
                "messages": messages,
                "current_snapshot": current_snapshot,
                "last_call": new_last_call,
                "repeats": new_repeats,
                "success_delta": 0,
                "failure_delta": 0,
                "decision": decision,
            }

        try:
            ok, text = await execute_tool(session, tools_by_name, action, args_dict)

            # Determine context for next turn
            # If agent explicitly asked for snapshot, give it all. Otherwise, use RAG.
            if ok and action == "browser_snapshot":
                logger.info("📸 Full snapshot requested by agent.")
                current_snapshot = text  # Update snapshot from tool output
                feedback = f"Action Result (OK):\n{text}"
            elif ok:
                logger.info("📸 Refreshing snapshot after action...")
                current_snapshot = await _wait_for_stable_snapshot(session)

                # Extract current URL from snapshot for better query generation
                current_url = "unknown"
                for line in current_snapshot.split("\n"):
                    if "Page URL:" in line:
                        current_url = line.split("Page URL:")[1].strip()
                        break

                # Generate optimized search query for next context
                next_query = await _generate_search_query(goal, thought, current_url)
                rag_context = _get_rag_context(
                    query=next_query, snapshot=current_snapshot, top_k=3
                )

                feedback = f"Action Result (OK):\n{text}\n\nRelevant Page Context:\n{rag_context}"
            else:
                # Action failed
                current_url = "unknown"
                for line in current_snapshot.split("\n"):
                    if "Page URL:" in line:
                        current_url = line.split("Page URL:")[1].strip()
                        break

                next_query = await _generate_search_query(goal, thought, current_url)
                rag_context = _get_rag_context(
                    query=next_query, snapshot=current_snapshot, top_k=3
                )

                logger.warning("❌ Action FAILED: %s", text[:100])
                feedback = (
                    f"Action FAILED: {text}\n"
                    f"Fix the arguments and try again.\n"
                    f"Remember to use [ref=ID] tags from the snapshot.\n"
                    f"Relevant Page Context:\n{rag_context}"
                )

            if ok:
                step_span.set_attribute("agent.step.outcome", "success")
                logger.info("✅ Action OK")
                success_delta = 1
                failure_delta = 0
            else:
                step_span.set_attribute("agent.step.outcome", "failed")
                success_delta = 0
                failure_delta = 1

        except Exception as e:
            step_span.set_attribute("agent.step.outcome", "exception")
            step_span.set_attribute("agent.step.error", str(e)[:500])
            if "Session terminated" in str(e):
                logger.error(
                    "💥 MCP session is gone (server restarted/crashed?). Stopping."
                )
                return {
                    "should_break": True,
                    "break_reason": "session_terminated",
                    "messages": messages,
                    "current_snapshot": current_snapshot,
                    "last_call": new_last_call,
                    "repeats": new_repeats,
                    "success_delta": 0,
                    "failure_delta": 1,
                    "decision": decision,
                }

            logger.error("⚠️ Tool execution error: %s", e)
            ok, text = False, str(e)

            current_url = "unknown"
            for line in current_snapshot.split("\n"):
                if "Page URL:" in line:
                    current_url = line.split("Page URL:")[1].strip()
                    break

            next_query = await _generate_search_query(goal, thought, current_url)
            rag_context = _get_rag_context(
                query=next_query, snapshot=current_snapshot, top_k=3
            )

            feedback = f"Action FAILED: {text}\nRelevant Page Context:\n{rag_context}"
            success_delta = 0
            failure_delta = 1

        messages.append({"role": "assistant", "content": llm_response})
        messages.append({"role": "user", "content": feedback})

        return {
            "should_break": False,
            "messages": messages,
            "current_snapshot": current_snapshot,
            "last_call": new_last_call,
            "repeats": new_repeats,
            "success_delta": success_delta,
            "failure_delta": failure_delta,
            "decision": decision,
        }


@chain(name="web-surfer-session")
async def _run_session(read_stream, write_stream, goal: str, args=None):
    """
    Execute the web surfer session with full telemetry coverage.
    Handles initialization, navigation, and the main reasoning loop with proper nesting.
    """
    tracer = otel_trace.get_tracer(__name__)
    async with ClientSession(read_stream, write_stream) as session:
        (
            tools_by_name,
            system_prompt,
            messages,
            current_snapshot,
            session_span,
        ) = await _initialize_session(session, goal, args)

        from .config import DEFAULT_MAX_REPEATS, DEFAULT_MAX_STEPS

        max_steps = args.max_steps if args else DEFAULT_MAX_STEPS
        max_repeats = args.max_repeats if args else DEFAULT_MAX_REPEATS

        last_call = None
        repeats = 0
        step_count = 0
        success_count = 0
        failure_count = 0
        completed_answer = None
        break_reason = None
        last_decision = None

        for step in range(1, max_steps + 1):
            step_count = step
            result = await _execute_step(
                step=step,
                max_steps=max_steps,
                messages=messages,
                args=args,
                session=session,
                tools_by_name=tools_by_name,
                current_snapshot=current_snapshot,
                last_call=last_call,
                repeats=repeats,
                max_repeats=max_repeats,
                tracer=tracer,
                last_decision=last_decision,
                goal=goal,
            )

            messages = result["messages"]
            current_snapshot = result["current_snapshot"]
            last_call = result["last_call"]
            repeats = result["repeats"]
            success_count += result["success_delta"]
            failure_count += result["failure_delta"]

            # Update last_decision for next iteration's RAG query
            last_decision = result.get("decision")

            if result["should_break"]:
                break_reason = result.get("break_reason")
                if break_reason == "completed":
                    completed_answer = result.get("answer")
                break

        if session_span.is_recording():
            session_span.set_attribute("agent.steps_taken", step_count)
            session_span.set_attribute("agent.actions_successful", success_count)
            session_span.set_attribute("agent.actions_failed", failure_count)

            if break_reason == "completed":
                session_span.set_attribute("agent.completion.success", True)
                session_span.set_attribute("agent.completion.answer", completed_answer)
            elif break_reason:
                session_span.set_attribute("agent.completion.success", False)
                session_span.set_attribute("agent.failure.reason", break_reason)
            else:
                logger.warning("⏹️ Reached MAX_STEPS (%d) without finishing", max_steps)
                session_span.set_attribute("agent.completion.success", False)
                session_span.set_attribute("agent.failure.reason", "max_steps_exceeded")

        logger.info("🔒 Closing MCP session gracefully...")
        try:
            await asyncio.sleep(0.5)
        except Exception:
            pass
