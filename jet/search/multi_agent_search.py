# web_research_agent.py
import logging
import operator
import os
from collections.abc import Sequence
from typing import Annotated, Any, Literal, TypedDict

import httpx
from bs4 import BeautifulSoup
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import ChatResult
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing_extensions import TypedDict

console = Console()
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger("web_research_agent")


# -----------------------
# Custom LLM to tolerate null/missing choices
# -----------------------
class TolerantChatOpenAI(ChatOpenAI):
    """ChatOpenAI that tolerates null/missing 'choices' from llama.cpp server."""

    def _create_chat_result(self, response: dict[str, Any]) -> ChatResult:
        # Force choices to be list (empty if null/missing)
        choices = response.get("choices")
        if choices is None:
            response["choices"] = []
        elif not isinstance(choices, list):
            response["choices"] = []

        # If still empty, add minimal dummy to prevent downstream failures
        if not response["choices"]:
            response["choices"] = [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "stop",
                }
            ]

        return super()._create_chat_result(response)


# -----------------------
# Configuration
# -----------------------
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://searxng.local")  # use your own instance!
MAX_STEPS = 12
RELEVANCE_THRESHOLD = 0.42  # slightly lower because local embedders can be noisier

llm = TolerantChatOpenAI(
    base_url="http://shawn-pc.local:8080/v1",
    api_key="sk-no-key-required",
    model="ggml-model",  # usually ignored, but some builds want something here
    temperature=0.15,
)

embedder = OpenAIEmbeddings(
    base_url="http://shawn-pc.local:8081/v1", api_key="sk-no-key-required"
)


# -----------------------
# Tools
# -----------------------
@tool
def web_search(query: str, num_results: int = 8) -> str:
    """Search the web via SearxNG and return formatted results."""
    try:
        resp = httpx.get(
            f"{SEARXNG_URL}/search",
            params={"q": query, "format": "json", "pageno": 1},
            timeout=12.0,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])[:num_results]

        if not results:
            return "No results found."

        formatted = []
        for r in results:
            formatted.append(
                f"- {r.get('title', '')}\n  {r.get('url', '')}\n  {r.get('content', '')[:220]}..."
            )
        return "\n\n".join(formatted)
    except Exception as e:
        return f"Search error: {str(e)}"


@tool
def fetch_page(url: str) -> str:
    """Fetch and clean main text content from a webpage."""
    try:
        resp = httpx.get(url, timeout=15.0, follow_redirects=True)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Simple heuristic clean
        for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        return text[:12000]  # truncate for token safety
    except Exception as e:
        return f"Fetch error: {str(e)}"


@tool
def is_content_relevant(content: str, query: str) -> dict:
    """Compute semantic relevance score between content and original query."""
    if not content or len(content.strip()) < 40:
        return {"relevant": False, "score": 0.0, "reason": "Content too short/empty"}

    q_emb = embedder.encode(query, convert_to_tensor=True)
    c_emb = embedder.encode(content, convert_to_tensor=True)
    score = util.cos_sim(q_emb, c_emb).item()

    return {
        "relevant": score >= RELEVANCE_THRESHOLD,
        "score": round(float(score), 3),
        "reason": f"Cosine similarity: {score:.3f}",
    }


tools = [web_search, fetch_page, is_content_relevant]

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)


# -----------------------
# State
# -----------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    task: str
    collected_content: Annotated[
        list[dict], operator.add
    ]  # [{"url":, "snippet":, "relevance":}]
    visited_urls: Annotated[set[str], operator.or_]
    steps_taken: int
    final_answer: str
    status: Literal["working", "complete", "failed", "max_steps"]


# -----------------------
# Nodes
# -----------------------
def agent_node(state: AgentState):
    """Main reasoning + tool calling node"""
    task = state["task"]
    collected = "\n".join(
        [
            f"- {c['url']}: {c.get('snippet', '')[:120]} (rel:{c.get('relevance', 0):.2f})"
            for c in state.get("collected_content", [])
        ]
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a precise web research agent.
Task: {task}

Current collected relevant content:
{collected}

Visited URLs (do NOT revisit): {visited}

Steps taken: {steps}/{max}

Rules:
- Use tools to gather information.
- Only add HIGHLY relevant content.
- If confident you have enough → write FINAL ANSWER and stop.
- Be concise. Use semantic check before storing.""",
            ),
            MessagesPlaceholder("messages"),
        ]
    )

    chain = (
        {
            "task": lambda s: s["task"],
            "collected": lambda s: collected,
            "visited": lambda s: ", ".join(s.get("visited_urls", set())),
            "steps": lambda s: s.get("steps_taken", 0),
            "max": lambda _: MAX_STEPS,
            "messages": lambda s: s["messages"],
        }
        | prompt
        | llm_with_tools
    )

    response = chain.invoke(state)

    return {"messages": [response], "steps_taken": state.get("steps_taken", 0) + 1}


def tool_node(state: AgentState):
    """Execute selected tools"""
    last_msg = state["messages"][-1]
    tool_calls = last_msg.tool_calls or []

    results = []
    new_content = []
    new_visited = set()

    for tc in tool_calls:
        tool_name = tc["name"]
        args = tc["args"]

        if tool_name == "web_search":
            res = web_search.invoke(args)
            results.append(
                AIMessage(content=f"Search results:\n{res}", tool_call_id=tc["id"])
            )

        elif tool_name == "fetch_page":
            url = args["url"]
            if url in state.get("visited_urls", set()):
                results.append(
                    AIMessage(content=f"Already visited: {url}", tool_call_id=tc["id"])
                )
                continue

            content = fetch_page.invoke(args)
            new_visited.add(url)

            rel_result = is_content_relevant.invoke(
                {"content": content[:800], "query": state["task"]}
            )
            if rel_result["relevant"]:
                snippet = content[:300] + "..." if len(content) > 300 else content
                new_content.append(
                    {
                        "url": url,
                        "snippet": snippet,
                        "relevance": rel_result["score"],
                        "full_text": content,  # keep full for final
                    }
                )
                console.print(
                    f"[green]✓ Stored relevant page:[/green] {url} (score {rel_result['score']:.2f})"
                )

            results.append(
                AIMessage(
                    content=f"Page content (relevance {rel_result['score']:.2f}): {content[:600]}...",
                    tool_call_id=tc["id"],
                )
            )

        elif tool_name == "is_content_relevant":
            res = is_content_relevant.invoke(args)
            results.append(
                AIMessage(content=f"Relevance check: {res}", tool_call_id=tc["id"])
            )

    return {
        "messages": results,
        "collected_content": new_content,
        "visited_urls": new_visited,
    }


def decide_finish(state: AgentState):
    """Router: continue or end?"""
    last_msg = state["messages"][-1]

    if state.get("steps_taken", 0) >= MAX_STEPS:
        return "max_steps"

    if "FINAL ANSWER" in last_msg.content or state.get("status") == "complete":
        return END

    if last_msg.tool_calls:
        return "tools"

    return "agent"  # re-prompt if unclear


def summarize_final(state: AgentState):
    """Final synthesis node"""
    collected = state.get("collected_content", [])
    if not collected:
        return {"final_answer": "No relevant information found.", "status": "complete"}

    texts = [c["full_text"] for c in collected]
    urls = [c["url"] for c in collected]

    prompt = f"""Synthesize a concise, accurate final answer for:

Task: {state["task"]}

Sources:
{chr(10).join([f"- {u}" for u in urls])}

Key content excerpts:
{chr(10).join(texts[:3])}  # truncate

Write clear final answer:"""

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({})

    return {"final_answer": answer, "status": "complete"}


# -----------------------
# Graph
# -----------------------
workflow = StateGraph(state_schema=AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_node("summarize", summarize_final)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    decide_finish,
    {
        "tools": "tools",
        END: END,
        "max_steps": "summarize",
    },
)
workflow.add_edge("tools", "agent")
workflow.add_edge("summarize", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


# -----------------------
# Runner with rich UI
# -----------------------
def run_research_agent(task: str):
    config = {"configurable": {"thread_id": "web-research-1"}}

    initial_state = {
        "task": task,
        "messages": [HumanMessage(content=task)],
        "collected_content": [],
        "visited_urls": set(),
        "steps_taken": 0,
        "status": "working",
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task_prog = progress.add_task(
            f"[cyan]Researching: {task[:60]}...", total=MAX_STEPS
        )

        for event in app.stream(initial_state, config, stream_mode="values"):
            current_step = event.get("steps_taken", 0)
            progress.update(task_prog, completed=current_step)

            last_msg = event["messages"][-1]
            if isinstance(last_msg, AIMessage) and last_msg.content.strip():
                if "FINAL ANSWER" in last_msg.content:
                    console.print("\n[bold green]Final Answer[/bold green]")
                    console.print(last_msg.content.split("FINAL ANSWER")[1].strip())
                elif last_msg.tool_calls:
                    console.print(
                        f"[yellow]→ Calling tools: {', '.join(t['name'] for t in last_msg.tool_calls)}[/yellow]"
                    )

        final_state = app.get_state(config).values
        console.rule("Summary")
        console.print(f"[bold]Status:[/bold] {final_state['status']}")
        console.print(f"[bold]Steps:[/bold] {final_state['steps_taken']}")
        console.print(
            f"[bold]Collected pages:[/bold] {len(final_state.get('collected_content', []))}"
        )
        console.print("\n[bold blue]Final synthesized answer:[/bold]")
        console.print(final_state.get("final_answer", "No answer generated."))


if __name__ == "__main__":
    console.rule("Web Research Agent Demo (LangGraph + SearxNG)", style="cyan")
    task = console.input("[bold]Enter research task[/bold]: ")
    run_research_agent(task)
