from config import AtomConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from state.atom_state import AtomState
from tools.fetch_url_tool import create_fetch_url_tool
from tools.search_tool import create_search_tool

ATOM_SYSTEM_PROMPT = """You are an atomic fact-finding agent. Your job is to answer a specific, narrow question using web search and page reading.

RULES:
1. Search first to identify promising URLs, then fetch_url with a clear goal.
2. Cite sources by URL in your reasoning.
3. When you have sufficient evidence to answer definitively, respond with EXACTLY this JSON format and NOTHING else:
   {"answer": "<concise factual answer>", "sources": ["<url1>", "<url2>"], "confidence": <0.0-1.0>}
4. If after exhaustive search you cannot find the answer, respond with:
   {"answer": null, "sources": [], "confidence": 0.0, "reason": "<why not found>"}
5. Do NOT guess. Only report what is directly supported by fetched page content.
6. You have a limited number of steps. Be efficient."""


def build_atom_graph(config: AtomConfig):
    search_tool = create_search_tool(config)
    fetch_tool = create_fetch_url_tool(config)
    tools = [search_tool, fetch_tool]

    llm = ChatOpenAI(
        model=config.llm_model,
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
        temperature=0.0,
        max_tokens=4096,
    ).bind_tools(tools)

    tool_node = ToolNode(tools)

    async def agent_node(state: AtomState) -> dict:
        if state["step_count"] >= state["max_steps"]:
            return {
                "is_complete": True,
                "result": {
                    "answer": None,
                    "sources": [],
                    "confidence": 0.0,
                    "reason": f"Exceeded max steps ({state['max_steps']})",
                },
                "messages": [
                    {"role": "assistant", "content": "Step budget exhausted."}
                ],
            }

        response = await llm.ainvoke(state["messages"])
        return {
            "messages": [response],
            "step_count": state["step_count"] + 1,
        }

    async def check_completion(state: AtomState) -> str:
        if state.get("is_complete"):
            return "end"
        last_msg = state["messages"][-1]
        content = last_msg.content if hasattr(last_msg, "content") else ""
        if (
            isinstance(content, str)
            and '"answer"' in content
            and ('"sources"' in content)
        ):
            return "end"
        if last_msg.tool_calls:
            return "tools"
        return "agent"

    graph = StateGraph(AtomState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        check_completion,
        {
            "tools": "tools",
            "agent": "agent",
            "end": END,
        },
    )
    graph.add_edge("tools", "agent")

    return graph.compile()
