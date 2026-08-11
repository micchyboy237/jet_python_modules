import logging
import time

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from .cache import DedupCache
from .llm_client import LocalLLMClient
from .nodes import planner_node, searcher_node, should_recurse, synthesizer_node
from .retriever import LocalRetriever
from .state import SwarmState

logger = logging.getLogger("webswarm")


async def run_swarm(query: str) -> str:
    llm = LocalLLMClient()
    retriever = LocalRetriever()
    dedup = DedupCache()

    graph = StateGraph(SwarmState)

    # ✅ Async wrappers that properly await the node coroutines
    async def _plan(state: SwarmState) -> dict:
        return await planner_node(state, llm)

    async def _search(state: SwarmState) -> dict:
        return await searcher_node(state, llm, retriever, dedup)

    async def _synthesize(state: SwarmState) -> dict:
        return await synthesizer_node(state, llm, retriever)

    graph.add_node("plan", _plan)
    graph.add_node("search", _search)
    graph.add_node("synthesize", _synthesize)

    graph.set_entry_point("plan")
    graph.add_conditional_edges("plan", lambda _: "search", {"search": "search"})
    graph.add_conditional_edges(
        "search",
        lambda state: should_recurse(state, llm),
        {"search": "search", "plan": "plan", "synthesize": "synthesize"},
    )
    graph.add_edge("synthesize", END)

    app = graph.compile(checkpointer=MemorySaver())

    initial_state: SwarmState = {
        "query": query,
        "subtasks": [],
        "findings": [],
        "iteration": 0,
        "tokens_used": 0,
        "start_time": time.time(),
        "final_answer": None,
    }

    result = await app.ainvoke(
        initial_state, config={"configurable": {"thread_id": query[:50]}}
    )
    return result.get("final_answer", "No answer generated.")
