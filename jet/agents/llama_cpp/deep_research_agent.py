#!/usr/bin/env python3
"""
LangGraph Deep Research Agent (Jet Infrastructure Edition)
- Uses ChatLlamaCpp + SearXNGSearchResults from jet.adapters
- Full OpenTelemetry/Phoenix observability (matching crag_base/react_with_telemetry)
- Cyclic nested search with reflection and fan-out
"""

import argparse
import json
import operator
import os
import uuid
from typing import Annotated, Any, Dict, List, Literal, TypedDict

from dotenv import load_dotenv
from jet.adapters.langchain.chat_llama_cpp import ChatLlamaCpp
from jet.adapters.langchain.tools.searxng_search_tool import SearXNGSearchResults
from jet.adapters.llama_cpp.config import LLM_MODEL, PHOENIX_REST_API
from jet.logger import logger
from langgraph.graph import END, START, Send, StateGraph
from langgraph.graph.message import add_messages
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from phoenix.otel import HTTPSpanExporter
from pydantic import BaseModel, Field

load_dotenv()

# =============================================================================
# OBSERVABILITY SETUP (mirrors crag_base.py / react_with_telemetry.py)
# =============================================================================
PROJECT_NAME = "deep-research-agent"
_resource = Resource.create({"openinference.project.name": PROJECT_NAME})
_provider = TracerProvider(resource=_resource)
_exporter = HTTPSpanExporter(endpoint=f"{PHOENIX_REST_API}/traces")
_provider.add_span_processor(BatchSpanProcessor(_exporter))
trace.set_tracer_provider(_provider)
tracer = trace.get_tracer(__name__)

PII_PATTERNS = ["ssn", "password", "api_key", "secret", "token"]


def _redact(text: str) -> str:
    """Redact sensitive content from text for safe tracing."""
    if not isinstance(text, str):
        return str(text)
    lower = text.lower()
    for pattern in PII_PATTERNS:
        if pattern in lower:
            return "[REDACTED]"
    return text


def _extract_token_usage(response_obj: Any) -> Dict[str, int]:
    """Extract token usage from LangChain response if available."""
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    try:
        if hasattr(response_obj, "usage_metadata") and response_obj.usage_metadata:
            um = response_obj.usage_metadata
            usage["prompt_tokens"] = um.get("input_tokens", 0)
            usage["completion_tokens"] = um.get("output_tokens", 0)
            usage["total_tokens"] = um.get("total_tokens", 0)
        elif (
            hasattr(response_obj, "response_metadata")
            and response_obj.response_metadata
        ):
            rm = response_obj.response_metadata
            if "token_usage" in rm:
                tu = rm["token_usage"]
                usage["prompt_tokens"] = tu.get("prompt_tokens", 0)
                usage["completion_tokens"] = tu.get("completion_tokens", 0)
                usage["total_tokens"] = tu.get("total_tokens", 0)
    except Exception:
        pass
    return usage


def _set_llm_token_attrs(span: Any, usage: Dict[str, int]) -> None:
    """Set token count attributes on an LLM span."""
    span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, usage["prompt_tokens"])
    span.set_attribute(
        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, usage["completion_tokens"]
    )
    span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, usage["total_tokens"])


# =============================================================================
# STATE DEFINITIONS
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
    # Transient internal state (underscore prefix convention)
    _reflection: Any


class ReflectionOutput(BaseModel):
    """Structured output for the reflection node."""

    is_sufficient: bool = Field(
        description="Whether gathered info fully answers the query"
    )
    knowledge_gaps: List[str] = Field(
        description="Missing information that needs more search"
    )
    follow_up_queries: List[str] = Field(description="New search queries to fill gaps")


class QueryGenerationOutput(BaseModel):
    """Structured output for initial query generation."""

    queries: List[str] = Field(description="List of targeted web search queries")


# =============================================================================
# NODE FUNCTIONS WITH FULL OBSERVABILITY
# =============================================================================
def generate_initial_queries(state: OverallState, llm: ChatLlamaCpp) -> dict:
    """Generate initial search queries from the user's question."""
    user_query = state["messages"][-1].content

    with tracer.start_as_current_span(
        "deep_research.llm.generate_initial_queries",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
            SpanAttributes.LLM_MODEL_NAME: llm._model,
            SpanAttributes.INPUT_VALUE: _redact(user_query),
        },
    ) as span:
        prompt = (
            f"Given the following research question, generate 2-3 targeted web search queries.\n\n"
            f"Question: {user_query}\n\n"
            f"Respond ONLY with valid JSON in this EXACT format:\n"
            f'{{"queries": ["query1", "query2"]}}\n'
            f"Do NOT include any other fields or explanations."
        )
        response = llm.with_structured_output(
            QueryGenerationOutput, method="json_mode"
        ).invoke(prompt)
        queries = response.queries if response.queries else ["general search"]

        usage = _extract_token_usage(response)
        _set_llm_token_attrs(span, usage)
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps(queries))
        span.set_attribute("deep_research.initial_query_count", len(queries))

        logger.info(f"Generated {len(queries)} initial queries: {queries}")
        return {"search_queries": queries}


def web_search(state: dict, search_tool: SearXNGSearchResults) -> dict:
    """Execute web search for a single query. Receives Send() state."""
    query = state.get("search_query", "")

    with tracer.start_as_current_span(
        "deep_research.tool.web_search",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.TOOL.value,
            SpanAttributes.TOOL_NAME: "searxng_search",
            SpanAttributes.INPUT_VALUE: _redact(query),
        },
    ) as span:
        result = search_tool.invoke({"query": query})

        # SearXNGSearchResults returns (formatted_string, raw_results) tuple
        if isinstance(result, tuple):
            formatted_content, raw_results = result
        else:
            formatted_content = result
            raw_results = []

        content_parts = []
        source_urls = []
        for r in raw_results:
            url = r.get("url", "")
            content = r.get("content", "")[:2000]
            title = r.get("title", "Untitled")
            score = r.get("score", "N/A")
            content_parts.append(
                f"[Source: {url} | Score: {score}]\n{title}\n{content}"
            )
            source_urls.append(url)

        combined = (
            "\n\n---\n\n".join(content_parts) if content_parts else formatted_content
        )

        span.set_attribute(SpanAttributes.OUTPUT_VALUE, _redact(combined[:3000]))
        span.set_attribute("deep_research.search_result_count", len(raw_results))
        span.set_attribute("deep_research.source_urls", json.dumps(source_urls))

        logger.info(f"Web search for '{query}' returned {len(raw_results)} results")
        return {
            "research_results": [combined],
            "sources": source_urls,
        }


def reflect_on_research(state: OverallState, llm: ChatLlamaCpp) -> dict:
    """Evaluate if current research is sufficient or needs deeper search."""
    all_results = "\n\n===\n\n".join(state.get("research_results", [])[-5:])
    original_query = state["messages"][-1].content

    with tracer.start_as_current_span(
        "deep_research.llm.reflect_on_research",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
            SpanAttributes.LLM_MODEL_NAME: llm._model,
            SpanAttributes.INPUT_VALUE: _redact(
                f"Query: {original_query}\nResults length: {len(all_results)}"
            ),
            "deep_research.loop_count": state.get("loop_count", 0),
        },
    ) as span:
        prompt = (
            f"Original Question: {original_query}\n\n"
            f"Research Gathered So Far:\n{all_results[:8000]}\n\n"
            f"Reflect on whether this research FULLY and ACCURATELY answers the original question.\n"
            f"If not, identify specific knowledge gaps and generate follow-up search queries.\n\n"
            f"Respond ONLY with valid JSON in this EXACT format:\n"
            f'{{"is_sufficient": false, "knowledge_gaps": ["gap1"], "follow_up_queries": ["query1"]}}\n'
            f"Do NOT include any other fields or explanations."
        )
        reflection = llm.with_structured_output(
            ReflectionOutput, method="json_mode"
        ).invoke(prompt)

        new_loop_count = state.get("loop_count", 0) + 1

        usage = _extract_token_usage(reflection)
        _set_llm_token_attrs(span, usage)
        span.set_attribute(
            SpanAttributes.OUTPUT_VALUE, json.dumps(reflection.model_dump())
        )
        span.set_attribute("deep_research.is_sufficient", reflection.is_sufficient)
        span.set_attribute(
            "deep_research.knowledge_gap_count", len(reflection.knowledge_gaps)
        )
        span.set_attribute(
            "deep_research.follow_up_query_count", len(reflection.follow_up_queries)
        )
        span.set_attribute("deep_research.new_loop_count", new_loop_count)

        logger.info(
            f"Reflection (loop {new_loop_count}): sufficient={reflection.is_sufficient}, "
            f"gaps={len(reflection.knowledge_gaps)}, follow_ups={len(reflection.follow_up_queries)}"
        )
        return {
            "loop_count": new_loop_count,
            "_reflection": reflection,
        }


def route_research(
    state: OverallState,
) -> Literal["finalize_report", "__continue_search__"]:
    """Conditional router: continue nested search or finalize."""
    reflection = state.get("_reflection")
    max_loops = state.get("max_loops", 3)
    loop_count = state.get("loop_count", 0)

    with tracer.start_as_current_span(
        "deep_research.route_research",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
            "deep_research.loop_count": loop_count,
            "deep_research.max_loops": max_loops,
            "deep_research.is_sufficient": reflection.is_sufficient
            if reflection
            else None,
        },
    ) as span:
        if reflection and (reflection.is_sufficient or loop_count >= max_loops):
            decision = "finalize_report"
        else:
            decision = "__continue_search__"

        span.set_attribute("deep_research.routing_decision", decision)
        logger.info(f"Routing decision: {decision}")
        return decision


def dispatch_follow_ups(state: OverallState) -> List[Send]:
    """Fan-out: create parallel Send objects for each follow-up query."""
    reflection = state.get("_reflection")
    if not reflection or not reflection.follow_up_queries:
        return []

    sends = [
        Send("web_search", {"search_query": q})
        for q in reflection.follow_up_queries[:3]
    ]

    logger.info(
        f"Dispatching {len(sends)} follow-up searches: {[s.kwargs['search_query'] for s in sends]}"
    )
    return sends


def finalize_report(state: OverallState, llm: ChatLlamaCpp) -> dict:
    """Synthesize all research into a cited final report."""
    all_results = "\n\n===\n\n".join(state.get("research_results", []))
    sources = list(set(state.get("sources", [])))
    original_query = state["messages"][-1].content

    with tracer.start_as_current_span(
        "deep_research.llm.finalize_report",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
            SpanAttributes.LLM_MODEL_NAME: llm._model,
            SpanAttributes.INPUT_VALUE: _redact(
                f"Query: {original_query}\nSources: {len(sources)}\nContent length: {len(all_results)}"
            ),
            "deep_research.total_source_count": len(sources),
            "deep_research.total_loop_count": state.get("loop_count", 0),
        },
    ) as span:
        sources_text = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sources))
        prompt = (
            f'Write a comprehensive, accurate research report answering:\n"{original_query}"\n\n'
            f"Use ONLY the following research context. Cite sources using [1], [2] etc.\n"
            f"Include a Sources section at the end listing all URLs.\n\n"
            f"Research Context:\n{all_results[:12000]}\n\n"
            f"Sources:\n{sources_text}"
        )
        report_msg = llm.invoke(prompt)
        report_content = report_msg.content

        usage = _extract_token_usage(report_msg)
        _set_llm_token_attrs(span, usage)
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, _redact(report_content[:3000]))
        span.set_attribute("deep_research.report_length", len(report_content))

        logger.success(f"Final report generated ({len(report_content)} chars)")
        return {"final_report": report_content}


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================
def build_graph(
    llm: ChatLlamaCpp, search_tool: SearXNGSearchResults, max_loops: int = 3
):
    """Build and compile the deep research LangGraph."""
    builder = StateGraph(OverallState)

    # Bind dependencies to node functions via closures
    builder.add_node(
        "generate_initial_queries", lambda state: generate_initial_queries(state, llm)
    )
    builder.add_node("web_search", lambda state: web_search(state, search_tool))
    builder.add_node(
        "reflect_on_research", lambda state: reflect_on_research(state, llm)
    )
    builder.add_node("dispatch_follow_ups", dispatch_follow_ups)
    builder.add_node("finalize_report", lambda state: finalize_report(state, llm))

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
                state.get("_reflection")
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
# CLI ENTRYPOINT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="LangGraph Deep Research Agent with SearXNG + Jet observability"
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
        default=LLM_MODEL,
        help=f"LLM model name (default: {LLM_MODEL})",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.3, help="LLM temperature (default: 0.3)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Max tokens per LLM call (default: 8192)",
    )
    parser.add_argument(
        "--search-count",
        type=int,
        default=5,
        help="Max search results per query (default: 5)",
    )
    parser.add_argument(
        "--search-engines",
        nargs="*",
        default=None,
        help="SearXNG engines to use (e.g., google bing duckduckgo)",
    )
    parser.add_argument(
        "--search-categories",
        nargs="*",
        default=["general"],
        help="SearXNG categories (default: general)",
    )
    parser.add_argument(
        "--searxng-url",
        type=str,
        default=os.getenv("SEARXNG_URL", "http://localhost:8888"),
        help="SearXNG instance URL (default: $SEARXNG_URL or http://localhost:8888)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print intermediate node execution status",
    )
    args = parser.parse_args()

    # Initialize Jet LLM (ChatLlamaCpp with built-in logging)
    llm = ChatLlamaCpp(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        agent_name="deep-research-agent",
        verbose=args.verbose,
    )

    # Initialize SearXNG search tool
    search_tool = SearXNGSearchResults(
        max_results=args.search_count,
        default_engines=args.search_engines,
        default_categories=args.search_categories,
        query_url=args.searxng_url,
        output_format="list",
    )

    session_id = str(uuid.uuid4())
    print(f"🔍 Starting deep research: '{args.query}'")
    print(
        f"   Model: {args.model} | Max loops: {args.max_loops} | Search: {args.search_count} results/query"
    )
    print(
        f"   SearXNG: {args.searxng_url} | Engines: {args.search_engines or 'default'}"
    )
    print(f"   Session: {session_id}\n")

    graph = build_graph(llm=llm, search_tool=search_tool, max_loops=args.max_loops)

    initial_state: OverallState = {
        "messages": [{"role": "user", "content": args.query}],
        "search_queries": [],
        "research_results": [],
        "sources": [],
        "loop_count": 0,
        "max_loops": args.max_loops,
        "final_report": "",
        "_reflection": None,
    }

    config = {"recursion_limit": args.max_loops * 10 + 20}

    # Execute with full tracing
    with tracer.start_as_current_span(
        "deep_research.session",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.AGENT.value,
            SpanAttributes.SESSION_ID: session_id,
            SpanAttributes.INPUT_VALUE: _redact(args.query),
            "deep_research.model": args.model,
            "deep_research.max_loops": args.max_loops,
            "deep_research.search_engines": json.dumps(args.search_engines or []),
            "deep_research.searxng_url": args.searxng_url,
        },
    ) as root_span:
        final_report = ""
        for event in graph.stream(initial_state, config=config, stream_mode="updates"):
            node_name = next(iter(event.keys()))
            if args.verbose:
                print(f"  ⚙️  [{node_name}] completed")
            # Capture final report from stream
            node_data = event[node_name]
            if isinstance(node_data, dict) and node_data.get("final_report"):
                final_report = node_data["final_report"]

        # Fallback: invoke directly if streaming didn't capture report
        if not final_report:
            result = graph.invoke(initial_state, config=config)
            final_report = result.get("final_report", "No report generated.")

        root_span.set_attribute(
            SpanAttributes.OUTPUT_VALUE, _redact(final_report[:3000])
        )
        root_span.set_attribute("deep_research.final_report_length", len(final_report))

    # Print trace link
    phoenix_host = PHOENIX_REST_API.rstrip("/")
    if phoenix_host.endswith("/v1"):
        phoenix_host = phoenix_host[:-3]
    trace_id_hex = format(root_span.get_span_context().trace_id, "032x")
    trace_url = f"{phoenix_host}/redirects/traces/{trace_id_hex}"

    print("\n" + "=" * 70)
    print("📋 FINAL RESEARCH REPORT")
    print("=" * 70)
    print(final_report)
    print("=" * 70)
    print(f"\n🔗 Trace: {trace_url}")
    print(f"   Session ID: {session_id}")


if __name__ == "__main__":
    main()
