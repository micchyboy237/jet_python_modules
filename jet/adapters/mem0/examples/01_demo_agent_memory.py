import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydantic")


from jet.adapters.mem0.agent_memory import (
    PHOENIX_URL,
    AgentState,
    console,
    generate_response,
    retrieve_agent_memory,
    retrieve_shared_memory,
    tracer,
    update_memory,
)
from langgraph.graph import END, StateGraph
from opentelemetry.trace import Status, StatusCode


# ---------------------------------------------------------------------------
# Graph Compilation
# ---------------------------------------------------------------------------
def compile_graph():
    graph = StateGraph(AgentState)
    graph.add_node("retrieve_shared", retrieve_shared_memory)
    graph.add_node("retrieve_agent", retrieve_agent_memory)
    graph.add_node("generate", generate_response)
    graph.add_node("update_memory", update_memory)
    graph.set_entry_point("retrieve_shared")
    graph.add_edge("retrieve_shared", "retrieve_agent")
    graph.add_edge("retrieve_agent", "generate")
    graph.add_edge("generate", "update_memory")
    graph.add_edge("update_memory", END)
    app = graph.compile()
    return app


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    console.print(f"🔍 Open [link={PHOENIX_URL}]{PHOENIX_URL}[/link] to view traces")
    console.print("=" * 60)

    app = compile_graph()

    test_cases = [
        {
            "messages": [{"role": "user", "content": "I prefer TypeScript examples"}],
            "user_id": "alex",
            "agent_id": "coder",
        },
        {
            "messages": [
                {"role": "user", "content": "Debug the React hook in auth.ts"}
            ],
            "user_id": "alex",
            "agent_id": "coder",
        },
        {
            "messages": [
                {"role": "user", "content": "What language should I use for docs?"}
            ],
            "user_id": "alex",
            "agent_id": "writer",
        },
    ]

    for i, tc in enumerate(test_cases, 1):
        with tracer.start_as_current_span(
            f"test_case_{i}", attributes={"agent_id": tc["agent_id"]}
        ) as span:
            trace_id = span.get_span_context().trace_id
            trace_url = (
                f"{PHOENIX_URL.rstrip('/')}/redirects/traces/{format(trace_id, '032x')}"
            )
            console.print(f"\n🔗 Test {i} trace: [link={trace_url}]{trace_url}[/link]")

            result = app.invoke({**tc, "shared_context": "", "agent_context": ""})
            agent_label = tc["agent_id"].capitalize()
            console.print(
                f"[bold green]{agent_label}:[/bold green] "
                f"{result['messages'][-1].content}\n"
            )

            span.set_status(Status(StatusCode.OK))

    console.print(
        "\n✅ Check Phoenix UI → '[bold]mem0-langgraph-dual-scope[/bold]' project"
    )
