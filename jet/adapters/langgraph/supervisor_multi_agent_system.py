"""
Production Supervisor Multi-Agent System
- Workers iterate internally with self-evaluation until subtask is complete
- Memory consolidation at worker-supervisor boundary prevents token overflow
- Full OpenTelemetry/Phoenix observability preserved from original codebase
- Safe message rendering, role-boundary enforcement, alternating-turn guarantee,
  synthetic bridge messages for attribution preservation, and coder originality constraint
"""

import os
import time
import warnings
from typing import Annotated, Literal, TypedDict

warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydantic")

import logging

from jet.adapters.langchain.factory import get_chat_openai
from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.mem0.factory import get_memory_config
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent
from mem0 import Memory
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from phoenix.otel import register
from rich.console import Console
from rich.logging import RichHandler

# ─── Observability Setup ─────────────────────────────────────────────────────
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, markup=True, rich_tracebacks=True)],
)
logger = logging.getLogger("supervisor-multi-agent")

PHOENIX_URL = os.getenv("LLM_OBS_PHOENIX_URL", "http://localhost:6006")


def setup_observability(project_name: str = "supervisor-multi-agent"):
    """Configure OpenTelemetry following chat_stream_observability pattern."""
    os.environ.setdefault(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_AND_EVENT"
    )
    tracer_provider = register(
        project_name=project_name, endpoint=f"{PHOENIX_URL}/v1/traces", batch=False
    )
    console.print(f"🔭 Observability → [link={PHOENIX_URL}]{PHOENIX_URL}[/link]")
    return tracer_provider


tracer_provider = setup_observability()
tracer = trace.get_tracer(__name__)

# ─── Mem0 + LLM Setup ────────────────────────────────────────────────────────
MEM0_CONFIG = get_memory_config("supervisor_multi_agent_system_v1")
memory = Memory.from_config(MEM0_CONFIG)


def get_llm(temperature: float = 0):
    """Factory wrapper matching existing jet adapter pattern."""
    return get_chat_openai(model=LLM_MODEL, temperature=temperature)


# ─── State Definition ────────────────────────────────────────────────────────
class SupervisorState(TypedDict):
    messages: Annotated[list, lambda x, y: x + y]
    next_agent: str
    user_id: str
    consolidated_context: str


# ─── Memory Consolidation ────────────────────────────────────────────────────
CONSOLIDATION_PROMPT = """Compress the following conversation history into a concise factual summary.
Preserve: key decisions, confirmed facts, completed subtasks, and open questions.
Discard: intermediate tool calls, verbose reasoning, and redundant confirmations.
Output ONLY the compressed summary in bullet points. Max 300 words."""

consolidation_llm = get_llm(temperature=0)


def _safe_role(msg: BaseMessage) -> str:
    """Extract role name safely from any message type."""
    return getattr(msg, "name", None) or getattr(msg, "type", None) or "unknown"


def consolidate_messages(messages: list[BaseMessage], max_raw_messages: int = 8) -> str:
    """Summarize older messages to prevent token overflow. Keeps recent messages verbatim."""
    if len(messages) <= max_raw_messages:
        return ""
    older = messages[:-max_raw_messages]
    formatted = "\n".join(
        f"[{_safe_role(m)}]: {str(m.content)[:500]}" for m in older if m.content
    )
    response = consolidation_llm.invoke(
        [
            SystemMessage(content=CONSOLIDATION_PROMPT),
            HumanMessage(content=formatted),
        ]
    )
    return response.content.strip()


# ─── Alternating-Turn Enforcement with Attribution Preservation ──────────────
def _ensure_alternating_turns(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Insert synthetic user turns between consecutive AI messages.

    Preserves agent attribution while satisfying LLM alternating-turn requirements.
    Unlike merging, this keeps each agent's output as a separate AIMessage with
    correct .name for display purposes.
    """
    if not messages:
        return messages

    result: list[BaseMessage] = [messages[0]]
    for msg in messages[1:]:
        prev = result[-1]
        if isinstance(prev, AIMessage) and isinstance(msg, AIMessage):
            prev_name = getattr(prev, "name", None) or "assistant"
            result.append(
                HumanMessage(
                    content=f"[Continuing from {prev_name}]",
                    name="system_bridge",
                )
            )
        result.append(msg)
    return result


# ─── Worker Agents with Role-Boundary Enforcement ────────────────────────────
RESEARCHER_PROMPT = """You are a research specialist. Your goal is to return a COMPLETE, VERIFIED answer.
Workflow:
1. Search for information relevant to the assigned subtask.
2. Evaluate: Does the result fully answer the subtask? Are sources consistent?
3. If incomplete or conflicting, refine your query and search again (max 3 iterations).
4. Only when satisfied, synthesize a final concise answer.

CRITICAL ROLE BOUNDARY: NEVER write code, scripts, or implementation examples.
If the task requires code, return ONLY the research findings and explicitly state
that coding is needed. The supervisor will delegate coding to the coder agent.

NEVER return partial results. NEVER ask the supervisor for help — iterate internally."""

CODER_PROMPT = """You are a coding specialist. Your goal is to return WORKING, TESTED code.
Workflow:
1. Write code to solve the assigned subtask.
2. Execute and verify output matches expected behavior.
3. If errors or incorrect output, debug and re-execute (max 3 iterations).
4. Only when code runs correctly, return the final solution with brief explanation.

CRITICAL ROLE BOUNDARY: NEVER perform web searches or general research.
Assume all necessary context has been provided by the supervisor.
If you lack critical information, state what is missing — do not attempt to research it yourself.

CRITICAL ORIGINALITY: Your output MUST be your own original code. Do NOT copy, repeat,
or echo any code or text from previous messages. Use prior context as reference only.
Produce a fresh, improved implementation based on the research findings provided.

NEVER return untested code. NEVER ask the supervisor for debugging help — iterate internally."""

researcher = create_react_agent(get_llm(), tools=[], prompt=RESEARCHER_PROMPT)
coder = create_react_agent(get_llm(), tools=[], prompt=CODER_PROMPT)


# ─── Worker Wrapper Nodes ────────────────────────────────────────────────────
def _worker_node(state: SupervisorState, agent_name: str, agent_graph) -> dict:
    """Generic worker wrapper: injects consolidated context, runs agent, compresses output."""
    with tracer.start_as_current_span(
        f"worker.{agent_name}", attributes={"user_id": state["user_id"]}
    ) as span:
        augmented_messages: list[BaseMessage] = []
        if state.get("consolidated_context"):
            augmented_messages.append(
                SystemMessage(
                    content=f"## Prior Context (summarized)\n{state['consolidated_context']}"
                )
            )
        augmented_messages.extend(state["messages"])

        # CRITICAL: Workers also receive multi-agent message history that can contain
        # consecutive AI messages. Apply alternating-turn enforcement with bridge messages.
        augmented_messages = _ensure_alternating_turns(augmented_messages)

        t0 = time.perf_counter()
        result = agent_graph.invoke({"messages": augmented_messages})
        duration = time.perf_counter() - t0

        final_content = str(result["messages"][-1].content)
        span.set_attribute("worker.duration_s", round(duration, 4))
        span.set_attribute("worker.output_length", len(final_content))
        span.set_status(Status(StatusCode.OK))
        logger.info(
            f"🤖 {agent_name}: completed in {duration:.3f}s ({len(final_content)} chars)"
        )

        return {
            "messages": [
                AIMessage(
                    content=f"[{agent_name.title()}]: {final_content}", name=agent_name
                )
            ]
        }


def researcher_node(state: SupervisorState) -> dict:
    return _worker_node(state, "researcher", researcher)


def coder_node(state: SupervisorState) -> dict:
    return _worker_node(state, "coder", coder)


# ─── Supervisor Node with Code-Leak Detection ────────────────────────────────
SUPERVISOR_PROMPT = """You are a supervisor managing two workers: 'researcher' and 'coder'.
Your job is to decompose the user request, route subtasks, and synthesize final answers.

Rules:
- Route ONE subtask at a time. Wait for worker response before routing next.
- When all subtasks are complete and you have enough information, respond with 'FINISH'.
- Use the consolidated context to avoid re-asking workers for already-completed work.
- Never do research or coding yourself. Always delegate.

Respond with ONLY: 'researcher', 'coder', or 'FINISH'."""

supervisor_llm = get_llm(temperature=0)

_CODE_INDICATORS = [
    "```python",
    "```javascript",
    "```typescript",
    "import asyncio",
    "def fetch_",
    "async def ",
]


def _researcher_produced_code(message_content: str) -> bool:
    """Check if researcher output contains code blocks despite role boundary."""
    content_lower = message_content.lower()
    return any(indicator.lower() in content_lower for indicator in _CODE_INDICATORS)


def supervisor_node(state: SupervisorState) -> dict:
    with tracer.start_as_current_span(
        "supervisor.route", attributes={"user_id": state["user_id"]}
    ) as span:
        consolidated = consolidate_messages(state["messages"])

        # Build routing messages with alternating-turn guarantee
        messages_for_routing: list[BaseMessage] = [
            SystemMessage(content=SUPERVISOR_PROMPT)
        ]
        if consolidated:
            messages_for_routing.append(
                SystemMessage(content=f"## Consolidated History\n{consolidated}")
            )
        messages_for_routing.extend(state["messages"][-4:])

        # CRITICAL: Ensure no two consecutive AI messages before sending to LLM
        messages_for_routing = _ensure_alternating_turns(messages_for_routing)

        response = supervisor_llm.invoke(messages_for_routing)
        next_agent = response.content.strip().lower()
        if next_agent not in ("researcher", "coder", "finish"):
            next_agent = "finish"

        # Guardrail: only check raw researcher messages, not merged/bridged ones.
        # Only re-route if coder hasn't already responded after this researcher output.
        if next_agent == "finish" and state["messages"]:
            last_raw_researcher = None
            for m in reversed(state["messages"]):
                if (
                    isinstance(m, AIMessage)
                    and getattr(m, "name", None) == "researcher"
                ):
                    last_raw_researcher = m
                    break
                if isinstance(m, AIMessage) and getattr(m, "name", None) == "coder":
                    break

            if last_raw_researcher and _researcher_produced_code(
                str(last_raw_researcher.content)
            ):
                original_query = (
                    str(state["messages"][0].content).lower()
                    if state["messages"]
                    else ""
                )
                coding_keywords = {
                    "write",
                    "code",
                    "implement",
                    "build",
                    "create",
                    "script",
                    "function",
                }
                coder_responded_after = any(
                    isinstance(m, AIMessage) and getattr(m, "name", None) == "coder"
                    for m in state["messages"][
                        state["messages"].index(last_raw_researcher) :
                    ]
                )
                if (
                    any(kw in original_query for kw in coding_keywords)
                    and not coder_responded_after
                ):
                    logger.warning(
                        "⚠️ Researcher produced code despite role boundary. "
                        "Re-routing to coder instead of finishing."
                    )
                    next_agent = "coder"

        span.set_attribute("supervisor.decision", next_agent)
        span.set_status(Status(StatusCode.OK))
        logger.info(f"🎯 Supervisor → {next_agent}")

        return {"next_agent": next_agent, "consolidated_context": consolidated}


# ─── Routing ─────────────────────────────────────────────────────────────────
def route_supervisor(
    state: SupervisorState,
) -> Literal["researcher", "coder", "__end__"]:
    decision = state.get("next_agent", "finish")
    if decision == "researcher":
        return "researcher"
    if decision == "coder":
        return "coder"
    return END


# ─── Graph Compilation ───────────────────────────────────────────────────────
graph = StateGraph(SupervisorState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("researcher", researcher_node)
graph.add_node("coder", coder_node)

graph.set_entry_point("supervisor")
graph.add_conditional_edges("supervisor", route_supervisor)
graph.add_edge("researcher", "supervisor")
graph.add_edge("coder", "supervisor")

app = graph.compile()

# ─── Main Execution ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    console.print("=" * 70)
    console.print(
        "🏗️  Supervisor Multi-Agent Demo (Iterative Workers + Memory Consolidation)"
    )
    console.print("=" * 70)

    test_query = (
        "Research the latest best practices for Python async error handling, "
        "then write a production-ready async HTTP client with retry logic using those practices."
    )

    with tracer.start_as_current_span(
        "demo_run", attributes={"user_id": "demo_user"}
    ) as span:
        result = app.invoke(
            {
                "messages": [HumanMessage(content=test_query)],
                "next_agent": "",
                "user_id": "demo_user",
                "consolidated_context": "",
            }
        )
        span.set_status(Status(StatusCode.OK))

    console.print("\n" + "=" * 70)
    console.print("[bold green]✅ Final Conversation:[/bold green]")
    for msg in result["messages"]:
        # Skip synthetic bridge messages in final display
        if getattr(msg, "name", None) == "system_bridge":
            continue
        role = _safe_role(msg).upper()
        content = str(msg.content)[:600] + (
            "..." if len(str(msg.content)) > 600 else ""
        )
        console.print(f"\n[bold cyan][{role}][/bold cyan]\n{content}")

    console.print(f"\n🔭 View full traces: [link={PHOENIX_URL}]{PHOENIX_URL}[/link]")
