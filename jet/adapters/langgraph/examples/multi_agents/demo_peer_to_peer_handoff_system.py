"""
Peer-to-Peer Handoff Multi-Agent System
- Agents use handoff tools to delegate tasks directly
- No supervisor; agents decide when to transfer control
- Supports cyclic delegation until task is complete
"""

from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent


# ─── Handoff Tools ───────────────────────────────────────────────────────────
@tool
def transfer_to_coder(task_description: str) -> str:
    """Transfer this conversation to the Coder agent. Use when the task requires writing, debugging, or executing Python code."""
    return f"Transferring to Coder: {task_description}"


@tool
def transfer_to_reviewer(task_description: str) -> str:
    """Transfer this conversation to the Code Reviewer agent. Use when code has been written and needs quality review, security audit, or improvement suggestions."""
    return f"Transferring to Reviewer: {task_description}"


@tool
def transfer_to_user(summary: str) -> str:
    """Transfer back to the user. Use when the task is fully complete or you need clarification."""
    return f"Task complete: {summary}"


# ─── Agent Definitions ──────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

coder_agent = create_react_agent(
    llm,
    tools=[transfer_to_reviewer, transfer_to_user],
    prompt=(
        "You are a senior Python developer. Write clean, production-quality code.\n"
        "When done, ALWAYS call transfer_to_reviewer to get your code reviewed.\n"
        "Only call transfer_to_user if the reviewer confirms the code is approved."
    ),
)

reviewer_agent = create_react_agent(
    llm,
    tools=[transfer_to_coder, transfer_to_user],
    prompt=(
        "You are a strict code reviewer. Evaluate code for correctness, security, and style.\n"
        "If issues exist, call transfer_to_coder with specific feedback.\n"
        "If code is approved, call transfer_to_user with a summary."
    ),
)


# ─── Active Agent Router ─────────────────────────────────────────────────────
class HandoffState(MessagesState):
    active_agent: str


def get_active_agent(state: HandoffState) -> Literal["coder", "reviewer", "__end__"]:
    """Route based on the most recent handoff tool call."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            tc_name = msg.tool_calls[0]["name"]
            if tc_name == "transfer_to_coder":
                return "coder"
            elif tc_name == "transfer_to_reviewer":
                return "reviewer"
            elif tc_name == "transfer_to_user":
                return END
    return "coder"  # Default entry point


# ─── Agent Wrapper Nodes ────────────────────────────────────────────────────
def coder_node(state: HandoffState) -> dict:
    result = coder_agent.invoke({"messages": state["messages"]})
    return {"messages": result["messages"], "active_agent": "coder"}


def reviewer_node(state: HandoffState) -> dict:
    result = reviewer_agent.invoke({"messages": state["messages"]})
    return {"messages": result["messages"], "active_agent": "reviewer"}


# ─── Build & Compile Graph ───────────────────────────────────────────────────
graph = StateGraph(HandoffState)
graph.add_node("coder", coder_node)
graph.add_node("reviewer", reviewer_node)

graph.set_conditional_entry_point(get_active_agent)
graph.add_conditional_edges("coder", get_active_agent)
graph.add_conditional_edges("reviewer", get_active_agent)

app = graph.compile()

# ─── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Write a Python function that safely parses JSON files with error handling, then get it reviewed."
                )
            ],
            "active_agent": "coder",
        }
    )
    for msg in result["messages"]:
        if isinstance(msg, HumanMessage):
            print(f"\n[USER]: {msg.content}")
        elif isinstance(msg, AIMessage):
            agent = getattr(msg, "name", "agent")
            content = msg.content or "(tool call)"
            print(f"\n[{agent.upper()}]: {content[:300]}")
