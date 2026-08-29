"""LangGraph Agentic RAG with Dynamic SearXNG Integration & Full Observability.
Reuses jet/adapters features:
- ChatLlamaCpp via get_chat_openai() for all LLM nodes
- Centralized config from jet.adapters.llama_cpp.config
- Automatic verbose logging via ChatLogger (no manual prints)
- Dynamic web retrieval via jet.search.searxng (replaces static vectorstore)
- Inline RAG prompt (no langchain.hub dependency)
- Full OpenTelemetry/OpenInference tracing to Phoenix
"""

import json
import sys
import uuid
from typing import Annotated, Literal, Sequence, TypedDict

from jet.adapters.langchain.factory import get_chat_openai
from jet.adapters.langchain.tools.searxng_search_tool import SearXNGSearchResults
from jet.adapters.llama_cpp.config import LLM_MODEL, PHOENIX_REST_API
from jet.logger import CustomLogger
from jet.logger.config import DEFAULT_LOGGER
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from phoenix.otel import HTTPSpanExporter
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Observability Setup
# ---------------------------------------------------------------------------
PROJECT_NAME = "langgraph-agentic-rag"
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


def _extract_token_usage(response_obj) -> dict:
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


def _set_llm_token_attrs(span, usage: dict):
    """Set token count attributes on an LLM span."""
    span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, usage["prompt_tokens"])
    span.set_attribute(
        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, usage["completion_tokens"]
    )
    span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, usage["total_tokens"])


def _print_trace_url(root_span):
    """Print Phoenix trace URL for the current session."""
    phoenix_host = PHOENIX_REST_API.rstrip("/")
    if phoenix_host.endswith("/v1"):
        phoenix_host = phoenix_host[:-3]
    trace_id_hex = format(root_span.get_span_context().trace_id, "032x")
    trace_url = f"{phoenix_host}/redirects/traces/{trace_id_hex}"
    logger.info("🔗 Trace: %s", trace_url)
    print(f"[Trace] {trace_url}")


# ---------------------------------------------------------------------------
# Core Initialization
# ---------------------------------------------------------------------------
llm = get_chat_openai(
    model=LLM_MODEL,
    temperature=0,
    streaming=True,
    verbose=True,
    enable_thinking=False,
    agent_name="agentic_rag",
)

logger = CustomLogger(DEFAULT_LOGGER, filename="agentic_rag_init.log")

searxng_tool = SearXNGSearchResults(
    max_results=5,
    categories=["general", "science", "news"],
    output_format="string",
)
tools = [searxng_tool]

logger.info(
    "Initialized agentic RAG with dynamic SearXNG tool | model=%s | max_results=%d",
    LLM_MODEL,
    searxng_tool.default_count,
)

AGENT_SYSTEM_PROMPT = (
    "You are a research assistant. For ANY question about current events, "
    "ongoing series, recent releases, or time-sensitive topics, you MUST "
    "use the searxng_search tool FIRST before answering. Never rely on "
    "your training data for 'today', 'currently', 'ongoing', 'latest', "
    "'new', or 'recent' queries. Only answer from your own knowledge if "
    "the question is purely historical or conceptual with no temporal component."
)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Use three sentences maximum and keep the answer concise.",
        ),
        ("human", "{question}\n\nContext:\n{context}"),
    ]
)


# ---------------------------------------------------------------------------
# Graph Nodes (with full observability)
# ---------------------------------------------------------------------------
def grade_documents(state: AgentState) -> Literal["generate", "rewrite"]:
    """Determines whether retrieved documents are relevant to the question."""

    class Grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    llm_with_tool = llm.with_structured_output(Grade)
    prompt = PromptTemplate(
        template=(
            "You are a grader assessing relevance of a retrieved document to a user question.\n\n"
            "Here is the retrieved document:\n\n{context}\n\n"
            "Here is the user question: {question}\n\n"
            "If the document contains keyword(s) or semantic meaning related to the user question, "
            "grade it as relevant.\n"
            "Give a binary score 'yes' or 'no' to indicate whether the document is relevant."
        ),
        input_variables=["context", "question"],
    )
    chain = prompt | llm_with_tool
    messages = state["messages"]
    last_message = messages[-1]
    question = messages[0].content
    docs = last_message.content

    with tracer.start_as_current_span(
        "agentic_rag.grade_documents",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
            SpanAttributes.INPUT_VALUE: _redact(
                f"Question: {question[:200]}\nDocs length: {len(docs)}"
            ),
        },
    ) as span:
        scored_result = chain.invoke({"question": question, "context": docs})
        usage = _extract_token_usage(scored_result)
        _set_llm_token_attrs(span, usage)
        span.set_attribute("agentic_rag.relevance_score", scored_result.binary_score)
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, scored_result.binary_score)

        logger.info(
            "grade_documents | question=%s | score=%s",
            question[:80],
            scored_result.binary_score,
        )

    if scored_result.binary_score == "yes":
        return "generate"
    return "rewrite"


def agent(state: AgentState) -> dict:
    """Invokes the agent model to decide retrieval or end."""
    messages = list(state["messages"])
    if not messages or messages[0].type != "system":
        messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)] + messages

    with tracer.start_as_current_span(
        "agentic_rag.agent",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
            SpanAttributes.LLM_MODEL_NAME: LLM_MODEL,
            SpanAttributes.INPUT_VALUE: _redact(
                json.dumps(
                    [
                        {"role": m.type, "content": m.content[:500]}
                        for m in messages[-3:]
                    ]
                )
            ),
        },
    ) as span:
        model_with_tools = llm.bind_tools(tools)
        response = model_with_tools.invoke(messages)
        usage = _extract_token_usage(response)
        _set_llm_token_attrs(span, usage)

        tool_calls = getattr(response, "tool_calls", []) or []
        span.set_attribute("agentic_rag.tool_call_count", len(tool_calls))
        span.set_attribute(
            SpanAttributes.OUTPUT_VALUE,
            _redact((response.content or "")[:2000]),
        )

        logger.info(
            "agent | tool_calls=%d | content_len=%d",
            len(tool_calls),
            len(response.content or ""),
        )

    return {"messages": [response]}


def retrieve(state: AgentState) -> dict:
    """Wraps ToolNode execution with TOOL-level observability."""
    with tracer.start_as_current_span(
        "agentic_rag.retrieve",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.TOOL.value,
            SpanAttributes.TOOL_NAME: "searxng_search",
        },
    ) as span:
        tool_node = ToolNode([searxng_tool])
        result = tool_node.invoke(state)

        # Extract retrieved content for span attributes
        new_messages = result.get("messages", [])
        if new_messages:
            last_content = new_messages[-1].content
            span.set_attribute(
                SpanAttributes.OUTPUT_VALUE, _redact(str(last_content)[:3000])
            )
            span.set_attribute(
                "agentic_rag.retrieved_content_length", len(str(last_content))
            )

        logger.info(
            "retrieve | messages_returned=%d",
            len(new_messages),
        )

    return result


def rewrite(state: AgentState) -> dict:
    """Transforms the query to produce a better question."""
    messages = state["messages"]
    question = messages[0].content
    msg = [
        HumanMessage(
            content=(
                "Look at the input and try to reason about the underlying semantic intent / meaning.\n\n"
                f"Here is the initial question:\n-------\n{question}\n-------\n\n"
                "Formulate an improved question:"
            )
        )
    ]

    with tracer.start_as_current_span(
        "agentic_rag.rewrite",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
            SpanAttributes.LLM_MODEL_NAME: LLM_MODEL,
            SpanAttributes.INPUT_VALUE: _redact(question[:1000]),
        },
    ) as span:
        response = llm.invoke(msg)
        usage = _extract_token_usage(response)
        _set_llm_token_attrs(span, usage)
        span.set_attribute(
            SpanAttributes.OUTPUT_VALUE, _redact((response.content or "")[:1000])
        )

        logger.info(
            "rewrite | original=%s | rewritten=%s",
            question[:80],
            (response.content or "")[:80],
        )

    return {"messages": [response]}


def generate(state: AgentState) -> dict:
    """Generates final answer from retrieved context."""
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]
    docs = last_message.content

    with tracer.start_as_current_span(
        "agentic_rag.generate",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
            SpanAttributes.LLM_MODEL_NAME: LLM_MODEL,
            SpanAttributes.INPUT_VALUE: _redact(
                f"Question: {question[:200]}\nContext length: {len(docs)}"
            ),
            "agentic_rag.context_length": len(docs),
        },
    ) as span:
        rag_chain = RAG_PROMPT | llm | StrOutputParser()
        response = rag_chain.invoke({"context": docs, "question": question})
        # StrOutputParser returns a string; no usage_metadata directly
        # Try to get usage from the last LLM call in the chain if possible
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, _redact(response[:3000]))
        span.set_attribute("agentic_rag.answer_length", len(response))

        logger.info("generate | answer_len=%d", len(response))

    return {"messages": [response]}


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent)
workflow.add_node("retrieve", retrieve)
workflow.add_node("rewrite", rewrite)
workflow.add_node("generate", generate)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {"tools": "retrieve", END: END},
)
workflow.add_conditional_edges("retrieve", grade_documents)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

graph = workflow.compile()
logger.info("LangGraph agentic RAG workflow compiled successfully")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pprint

    query = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What does Lilian Weng say about the types of agent memory?"
    )

    session_id = str(uuid.uuid4())
    logger.info("=== Starting agentic RAG run ===")
    logger.info("Query: %s", query)

    inputs = {"messages": [("user", query)]}

    with tracer.start_as_current_span(
        "agentic_rag.session",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.AGENT.value,
            SpanAttributes.SESSION_ID: session_id,
            SpanAttributes.INPUT_VALUE: _redact(query),
            "agentic_rag.query": _redact(query),
            "agentic_rag.model": LLM_MODEL,
        },
    ) as root_span:
        final_output = ""
        for output in graph.stream(inputs):
            for key, value in output.items():
                pprint.pprint(f"Output from node '{key}':")
                pprint.pprint("---")
                pprint.pprint(value, indent=2, width=80, depth=None)
                # Capture last generate output as final answer
                if key == "generate" and isinstance(value, dict):
                    msgs = value.get("messages", [])
                    if msgs:
                        final_output = (
                            msgs[-1].content
                            if hasattr(msgs[-1], "content")
                            else str(msgs[-1])
                        )
            pprint.pprint("\n---\n")

        root_span.set_attribute(
            SpanAttributes.OUTPUT_VALUE, _redact(final_output[:3000])
        )
        root_span.set_attribute("agentic_rag.session_id", session_id)

        _print_trace_url(root_span)

    logger.info("=== Agentic RAG run complete ===")
