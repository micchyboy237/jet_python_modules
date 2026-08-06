"""
Supervisor Multi-Agent System
- Supervisor routes tasks to Researcher or Coder subgraphs
- Each worker has dedicated tools and system prompts
- Supervisor synthesizes final answer from worker outputs
"""

import operator
from typing import Annotated, Literal, TypedDict

from jet.adapters.langchain.factory import get_chat_openai
from jet.adapters.llama_cpp.config import LLM_MODEL
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent


# ─── Shared State ────────────────────────────────────────────────────────────
class SupervisorState(TypedDict):
    messages: Annotated[list, operator.add]
    next_agent: str


# ─── Worker Agents (ReAct subgraphs with tools) ─────────────────────────────
def make_researcher():
    """Researcher agent with search tool."""
    from langchain_community.tools import DuckDuckGoSearchRun

    tools = [DuckDuckGoSearchRun()]
    return create_react_agent(
        get_chat_openai(model=LLM_MODEL, temperature=0),
        tools,
        prompt="You are a researcher. Find factual information and summarize findings concisely.",
    )


def make_coder():
    """Coder agent with Python REPL tool."""
    from langchain_experimental.tools import PythonREPLTool

    tools = [PythonREPLTool()]
    return create_react_agent(
        get_chat_openai(model=LLM_MODEL, temperature=0),
        tools,
        prompt="You are a Python programmer. Write and execute code to solve problems. Always print results.",
    )


researcher = make_researcher()
coder = make_coder()


# ─── Supervisor Node ─────────────────────────────────────────────────────────
SUPERVISOR_PROMPT = """You are a supervisor managing two workers: 'researcher' and 'coder'.
Given the user request, decide which worker should handle it.
Respond with ONLY one word: 'researcher', 'coder', or 'FINISH' (when you have enough info to answer directly)."""

supervisor_llm = get_chat_openai(model=LLM_MODEL, temperature=0)


def supervisor_node(state: SupervisorState) -> dict:
    response = supervisor_llm.invoke(
        [SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"]
    )
    next_agent = response.content.strip().lower()
    if next_agent not in ("researcher", "coder", "finish"):
        next_agent = "finish"
    return {"next_agent": next_agent}


# ─── Worker Wrapper Nodes ────────────────────────────────────────────────────
def researcher_node(state: SupervisorState) -> dict:
    result = researcher.invoke({"messages": state["messages"]})
    last_msg = result["messages"][-1]
    return {
        "messages": [
            AIMessage(content=f"[Researcher]: {last_msg.content}", name="researcher")
        ]
    }


def coder_node(state: SupervisorState) -> dict:
    result = coder.invoke({"messages": state["messages"]})
    last_msg = result["messages"][-1]
    return {
        "messages": [AIMessage(content=f"[Coder]: {last_msg.content}", name="coder")]
    }


# ─── Routing Logic ───────────────────────────────────────────────────────────
def route_supervisor(
    state: SupervisorState,
) -> Literal["researcher", "coder", "__end__"]:
    if state["next_agent"] == "researcher":
        return "researcher"
    elif state["next_agent"] == "coder":
        return "coder"
    return END


# ─── Build & Compile Graph ───────────────────────────────────────────────────
graph = StateGraph(SupervisorState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("researcher", researcher_node)
graph.add_node("coder", coder_node)

graph.set_entry_point("supervisor")
graph.add_conditional_edges("supervisor", route_supervisor)
graph.add_edge("researcher", "supervisor")  # Workers always report back to supervisor
graph.add_edge("coder", "supervisor")

app = graph.compile()

# ─── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content="What is the current population of Tokyo? Then write Python code to calculate how many years until it reaches 20 million at 0.5% annual growth."
                )
            ],
            "next_agent": "",
        }
    )
    for msg in result["messages"]:
        role = getattr(msg, "name", msg.type)
        print(f"\n{'=' * 60}\n[{role.upper()}]\n{msg.content}")
