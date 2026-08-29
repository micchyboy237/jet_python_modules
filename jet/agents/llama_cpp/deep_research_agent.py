#!/usr/bin/env python3
"""
LangGraph Deep Research Agent
Implements a cyclic web search agent with reflection and nested query generation.
"""

import argparse
import operator
import os
from typing import Annotated, List, Literal, TypedDict

from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, Send, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

load_dotenv()


# =============================================================================
# 1. STATE DEFINITIONS
# =============================================================================
class OverallState(TypedDict):
    """Central state shared across all nodes."""

    messages: Annotated[list, add_messages]
    search_queries: Annotated[List[str], operator.add]
    research_results: Annotated[List[str], operator.add]
    sources: Annotated[List[str], operator.add]
    loop_count: int
    max_loops: int
    final_report: str


class ReflectionOutput(BaseModel):
    """Structured output for the reflection node."""

    is_sufficient: bool = Field(
        description="Whether gathered info fully answers the query"
    )
    knowledge_gaps: List[str] = Field(
        description="Missing information that needs more search"
    )
    follow_up_queries: List[str] = Field(description="New search queries to fill gaps")


# =============================================================================
# 2. NODE FUNCTIONS
# =============================================================================
def generate_initial_queries(state: OverallState) -> dict:
    """Generate initial search queries from the user's question."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"""Given the following research question, generate 2-3 targeted web search queries.
    
Question: {state["messages"][-1].content}

Return ONLY a JSON object with key 'queries' containing a list of strings."""

    response = llm.with_structured_output(ReflectionOutput).invoke(prompt)
    return {
        "search_queries": response.follow_up_queries
        if response.follow_up_queries
        else ["general search"]
    }


def web_search(state: dict) -> dict:
    """Execute web search for a single query. Receives Send() state."""
    query = state.get("search_query", "")
    search_tool = TavilySearchResults(max_results=3, include_raw_content=True)

    results = search_tool.invoke(query)

    # Extract content and sources
    content_parts = []
    source_urls = []
    for r in results:
        content_parts.append(
            f"[Source: {r['url']}]\n{r.get('raw_content', r['content'])[:2000]}"
        )
        source_urls.append(r["url"])

    combined = "\n\n---\n\n".join(content_parts)
    return {
        "research_results": [combined],
        "sources": source_urls,
    }


def reflect_on_research(state: OverallState) -> dict:
    """Evaluate if current research is sufficient or needs deeper search."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    all_results = "\n\n===\n\n".join(
        state.get("research_results", [])[-5:]
    )  # Last 5 chunks
    original_query = state["messages"][-1].content

    prompt = f"""Original Question: {original_query}

Research Gathered So Far:
{all_results[:8000]}

Reflect on whether this research FULLY and ACCURATELY answers the original question.
If not, identify specific knowledge gaps and generate follow-up search queries."""

    reflection = llm.with_structured_output(ReflectionOutput).invoke(prompt)

    new_loop_count = state.get("loop_count", 0) + 1
    return {
        "loop_count": new_loop_count,
        "_reflection": reflection,  # Store temporarily for router
    }


def route_research(
    state: OverallState,
) -> Literal["finalize_report", "__continue_search__"]:
    """Conditional router: continue nested search or finalize."""
    reflection = state.get("_reflection")
    max_loops = state.get("max_loops", 3)
    loop_count = state.get("loop_count", 0)

    if reflection and (reflection.is_sufficient or loop_count >= max_loops):
        return "finalize_report"
    return "__continue_search__"


def dispatch_follow_ups(state: OverallState) -> List[Send]:
    """Fan-out: create parallel Send objects for each follow-up query."""
    reflection = state.get("_reflection")
    if not reflection or not reflection.follow_up_queries:
        return []

    return [
        Send("web_search", {"search_query": q})
        for q in reflection.follow_up_queries[:3]  # Cap parallel searches
    ]


def finalize_report(state: OverallState) -> dict:
    """Synthesize all research into a cited final report."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

    all_results = "\n\n===\n\n".join(state.get("research_results", []))
    sources = list(set(state.get("sources", [])))
    original_query = state["messages"][-1].content

    prompt = f"""Write a comprehensive, accurate research report answering:
"{original_query}"

Use ONLY the following research context. Cite sources using [1], [2] etc.
Include a Sources section at the end listing all URLs.

Research Context:
{all_results[:12000]}

Sources:
{chr(10).join(f"{i + 1}. {s}" for i, s in enumerate(sources))}"""

    report = llm.invoke(prompt)
    return {"final_report": report.content}


# =============================================================================
# 3. GRAPH CONSTRUCTION
# =============================================================================
def build_graph(max_loops: int = 3):
    builder = StateGraph(OverallState)

    # Add nodes
    builder.add_node("generate_initial_queries", generate_initial_queries)
    builder.add_node("web_search", web_search)
    builder.add_node("reflect_on_research", reflect_on_research)
    builder.add_node("dispatch_follow_ups", dispatch_follow_ups)
    builder.add_node("finalize_report", finalize_report)

    # Edges
    builder.add_edge(START, "generate_initial_queries")
    builder.add_conditional_edges(
        "generate_initial_queries",
        lambda state: [
            Send("web_search", {"search_query": q}) for q in state["search_queries"]
        ],
        ["web_search"],
    )
    builder.add_edge("web_search", "reflect_on_research")
    builder.add_conditional_edges(
        "reflect_on_research",
        route_research,
        {
            "finalize_report": "finalize_report",
            "__continue_search__": "dispatch_follow_ups",
        },
    )
    builder.add_conditional_edges(
        "dispatch_follow_ups",
        lambda state: [
            Send("web_search", {"search_query": q})
            for q in (
                state.get("_reflection", None)
                or ReflectionOutput(
                    is_sufficient=True, knowledge_gaps=[], follow_up_queries=[]
                )
            ).follow_up_queries[:3]
        ]
        or [END],
        ["web_search"],
    )
    builder.add_edge("finalize_report", END)

    return builder.compile()


# =============================================================================
# 4. CLI ENTRYPOINT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="LangGraph Deep Research Agent with nested web search"
    )
    parser.add_argument("query", type=str, help="Research question to investigate")
    parser.add_argument(
        "--max-loops",
        type=int,
        default=3,
        help="Maximum research depth/loops (default: 3)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model for search/reflection (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print intermediate states during execution",
    )
    args = parser.parse_args()

    # Validate API keys
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("ERROR: OPENAI_API_KEY not set. Export it or add to .env")
    if not os.getenv("TAVILY_API_KEY"):
        raise SystemExit(
            "ERROR: TAVILY_API_KEY not set. Get free key at https://tavily.com"
        )

    print(f"🔍 Starting deep research: '{args.query}'")
    print(f"   Max loops: {args.max_loops} | Model: {args.model}\n")

    graph = build_graph(max_loops=args.max_loops)

    initial_state = {
        "messages": [{"role": "user", "content": args.query}],
        "search_queries": [],
        "research_results": [],
        "sources": [],
        "loop_count": 0,
        "max_loops": args.max_loops,
        "final_report": "",
    }

    # Stream execution for visibility
    config = {"recursion_limit": args.max_loops * 10 + 20}
    final_state = None

    for event in graph.stream(initial_state, config=config, stream_mode="updates"):
        node_name = next(iter(event.keys()))
        if args.verbose:
            print(f"  ⚙️  [{node_name}] completed")
        final_state = event

    # Output final report
    report = ""
    for ev in graph.stream(initial_state, config=config, stream_mode="values"):
        if ev.get("final_report"):
            report = ev["final_report"]

    if not report:
        # Fallback: run without streaming
        result = graph.invoke(initial_state, config=config)
        report = result.get("final_report", "No report generated.")

    print("\n" + "=" * 70)
    print("📋 FINAL RESEARCH REPORT")
    print("=" * 70)
    print(report)
    print("=" * 70)


if __name__ == "__main__":
    main()
