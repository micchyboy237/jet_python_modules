# jet_python_modules/jet/adapters/langgraph/examples/rag/langgraph_crag.py
"""
Corrective RAG (CRAG) with LangGraph, Local Models, and Full Observability.

Features:
- Uses jet.adapters factories for LLM/Embeddings (ChatLlamaCpp, OpenAIEmbeddings)
- SearXNG search via jet.adapters.langchain.tools.searxng_search_tool
- Complete OpenTelemetry tracing to Phoenix (matching crag_base/react_with_telemetry)
- CLI interface with argparse and sensible defaults
"""

import argparse
import json
import os
import uuid
from typing import List, TypedDict

from jet.adapters.langchain.factory import get_chat_openai, get_openai_embeddings
from jet.adapters.langchain.tools.searxng_search_tool import SearXNGSearchResults
from jet.adapters.llama_cpp.config import (
    EMBED_MODEL_LG,
    LLM_MODEL,
    PHOENIX_REST_API,
)
from jet.logger import logger
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from phoenix.otel import HTTPSpanExporter
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Observability Setup (mirrors crag_base / react_with_telemetry)
# ---------------------------------------------------------------------------
PROJECT_NAME = "langgraph-crag-local"

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


# ---------------------------------------------------------------------------
# Graph State
# ---------------------------------------------------------------------------
class GraphState(TypedDict):
    """Represents the state of our graph."""

    question: str
    generation: str
    web_search: str
    documents: List[str]


# ---------------------------------------------------------------------------
# Grading / Rewriting Pydantic Models
# ---------------------------------------------------------------------------
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


# ---------------------------------------------------------------------------
# CRAG Graph Builder
# ---------------------------------------------------------------------------
def build_crag_graph(
    retriever,
    llm,
    search_tool: SearXNGSearchResults,
):
    """Build and compile the CRAG workflow with full observability."""

    # --- Manual JSON grader (reliable with small local LLMs) ---
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a grader assessing relevance of a retrieved document to a user question.\n"
                    "If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.\n\n"
                    "Respond ONLY with valid JSON in this EXACT format:\n"
                    '{{"binary_score": "yes"}} or {{"binary_score": "no"}}\n'
                    "Do NOT include any other fields, explanations, or markdown."
                ),
            ),
            (
                "human",
                "Retrieved document: \n\n {document} \n\n User question: {question}",
            ),
        ]
    )
    grade_chain = grade_prompt | llm

    # --- RAG generation chain ---
    rag_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an assistant for question-answering tasks. "
                    "Use the following pieces of retrieved context to answer the question. "
                    "If you don't know the answer, just say that you don't know. "
                    "Use three sentences maximum and keep the answer concise."
                ),
            ),
            ("human", "Context: {context}\n\nQuestion: {question}"),
        ]
    )
    rag_chain = rag_prompt | llm

    # --- Query rewriter ---
    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a question re-writer that converts an input question to a better version that is optimized\n"
                    "for web search. Look at the input and try to reason about the underlying semantic intent / meaning.\n\n"
                    "Respond ONLY with the rewritten question as plain text. Do NOT include JSON, explanations, or markdown."
                ),
            ),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question.",
            ),
        ]
    )
    question_rewriter = rewrite_prompt | llm

    # ---- Node functions with manual tracing ----

    def retrieve(state):
        with tracer.start_as_current_span(
            "crag.retrieve",
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.RETRIEVER.value,
                SpanAttributes.INPUT_VALUE: _redact(state["question"]),
            },
        ) as span:
            logger.info("---RETRIEVE---")
            question = state["question"]
            documents = retriever.invoke(question)
            span.set_attribute("crag.retrieved_doc_count", len(documents))
            for i, doc in enumerate(documents):
                span.set_attribute(
                    f"crag.retrieved_doc.{i}.content", _redact(doc.page_content[:2000])
                )
            return {"documents": documents, "question": question}

    def generate(state):
        with tracer.start_as_current_span(
            "crag.generate",
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
                SpanAttributes.LLM_MODEL_NAME: llm._model,
                SpanAttributes.INPUT_VALUE: _redact(state["question"]),
            },
        ) as span:
            logger.info("---GENERATE---")
            question = state["question"]
            documents = state["documents"]
            context = "\n\n".join(
                doc.page_content if hasattr(doc, "page_content") else str(doc)
                for doc in documents
            )
            response_msg = rag_chain.invoke({"context": context, "question": question})
            generation = (
                response_msg.content
                if hasattr(response_msg, "content")
                else str(response_msg)
            )
            usage = _extract_token_usage(response_msg)
            _set_llm_token_attrs(span, usage)
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, _redact(generation[:3000]))
            return {
                "documents": documents,
                "question": question,
                "generation": generation,
            }

    def grade_documents(state):
        with tracer.start_as_current_span(
            "crag.grade_documents",
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
                SpanAttributes.INPUT_VALUE: _redact(state["question"]),
                "crag.doc_count": len(state["documents"]),
            },
        ) as span:
            logger.info("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
            question = state["question"]
            documents = state["documents"]
            filtered_docs = []
            web_search = "No"
            for d in documents:
                content = d.page_content if hasattr(d, "page_content") else str(d)
                response_msg = grade_chain.invoke(
                    {"question": question, "document": content}
                )
                raw = (
                    response_msg.content.strip()
                    if hasattr(response_msg, "content")
                    else str(response_msg).strip()
                )
                try:
                    parsed = json.loads(raw)
                    grade = parsed.get("binary_score", "no").lower().strip()
                except (json.JSONDecodeError, AttributeError):
                    # Fallback: check if raw text contains yes/no
                    grade = "yes" if "yes" in raw.lower() else "no"
                    logger.warning(
                        f"JSON parse failed for grader, fallback to '{grade}'. Raw: {raw[:100]}"
                    )

                if grade == "yes":
                    logger.info("---GRADE: DOCUMENT RELEVANT---")
                    filtered_docs.append(d)
                else:
                    logger.info("---GRADE: DOCUMENT NOT RELEVANT---")
                    web_search = "Yes"
            span.set_attribute("crag.filtered_doc_count", len(filtered_docs))
            span.set_attribute("crag.web_search_needed", web_search)
            return {
                "documents": filtered_docs,
                "question": question,
                "web_search": web_search,
            }

    def transform_query(state):
        with tracer.start_as_current_span(
            "crag.transform_query",
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
                SpanAttributes.LLM_MODEL_NAME: llm._model,
                SpanAttributes.INPUT_VALUE: _redact(state["question"]),
            },
        ) as span:
            logger.info("---TRANSFORM QUERY---")
            question = state["question"]
            documents = state["documents"]
            response_msg = question_rewriter.invoke({"question": question})
            better_question = (
                response_msg.content
                if hasattr(response_msg, "content")
                else str(response_msg)
            )
            usage = _extract_token_usage(response_msg)
            _set_llm_token_attrs(span, usage)
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, _redact(better_question))
            return {"documents": documents, "question": better_question}

    def web_search_node(state):
        with tracer.start_as_current_span(
            "crag.web_search",
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.TOOL.value,
                SpanAttributes.TOOL_NAME: "searxng_search",
                SpanAttributes.INPUT_VALUE: _redact(state["question"]),
            },
        ) as span:
            logger.info("---WEB SEARCH (SearXNG)---")
            question = state["question"]
            documents = state["documents"]
            result = search_tool.invoke({"query": question})
            # SearXNGSearchResults returns (formatted_string, raw_results) via content_and_artifact
            if isinstance(result, tuple):
                web_content = result[0]
            else:
                web_content = str(result)
            span.set_attribute("crag.web_result_length", len(web_content))
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, _redact(web_content[:2000]))
            web_doc = Document(page_content=web_content)
            documents.append(web_doc)
            return {"documents": documents, "question": question}

    def decide_to_generate(state):
        logger.info("---ASSESS GRADED DOCUMENTS---")
        web_search = state["web_search"]
        if web_search == "Yes":
            logger.info(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
            )
            return "transform_query"
        else:
            logger.info("---DECISION: GENERATE---")
            return "generate"

    # ---- Build graph ----
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search_node", web_search_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"transform_query": "transform_query", "generate": "generate"},
    )
    workflow.add_edge("transform_query", "web_search_node")
    workflow.add_edge("web_search_node", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# Indexing Helper
# ---------------------------------------------------------------------------
def build_vectorstore(urls: List[str], embed_model: str, chunk_size: int = 250):
    """Load web docs, split, and index into Chroma with tracing."""
    with tracer.start_as_current_span(
        "crag.startup.build_vectorstore",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.EMBEDDING.value,
            "crag.urls": json.dumps(urls),
            "crag.chunk_size": chunk_size,
            SpanAttributes.EMBEDDING_MODEL_NAME: embed_model,
        },
    ) as span:
        logger.info(f"Loading {len(urls)} URLs into vector store...")
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]
        span.set_attribute("crag.raw_document_count", len(docs_list))

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=0
        )
        doc_splits = text_splitter.split_documents(docs_list)
        span.set_attribute("crag.chunk_count", len(doc_splits))

        embeddings = get_openai_embeddings(embed_model)
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name="rag-chroma",
            embedding=embeddings,
        )
        logger.success(f"Indexed {len(doc_splits)} chunks into Chroma")
        return vectorstore


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
DEFAULT_URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]


def main():
    parser = argparse.ArgumentParser(
        description="Run Corrective RAG (CRAG) with LangGraph and local models."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="What are the types of agent memory?",
        help="Question to ask the CRAG agent.",
    )
    parser.add_argument(
        "--model", default=LLM_MODEL, help=f"LLM model name (default: {LLM_MODEL})"
    )
    parser.add_argument(
        "--embed-model",
        default=EMBED_MODEL_LG,
        help=f"Embedding model (default: {EMBED_MODEL_LG})",
    )
    parser.add_argument(
        "--searxng-url",
        default=os.getenv("SEARXNG_URL", "http://localhost:8888"),
        help="SearXNG instance URL.",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=250, help="Text chunk size for splitting."
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=3,
        help="Max SearXNG search results per query.",
    )
    args = parser.parse_args()

    session_id = str(uuid.uuid4())
    logger.info(f"Session ID: {session_id}")
    logger.info(f"Query: {args.query}")

    with tracer.start_as_current_span(
        "crag.session",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.AGENT.value,
            SpanAttributes.SESSION_ID: session_id,
            SpanAttributes.INPUT_VALUE: _redact(args.query),
            "crag.model": args.model,
            "crag.embed_model": args.embed_model,
        },
    ) as root_span:
        # Build vector store
        vectorstore = build_vectorstore(
            urls=DEFAULT_URLS,
            embed_model=args.embed_model,
            chunk_size=args.chunk_size,
        )
        retriever = vectorstore.as_retriever()

        # Initialize LLM via jet factory
        llm = get_chat_openai(model=args.model, max_tokens=1000, temperature=0)

        # Initialize SearXNG search tool
        search_tool = SearXNGSearchResults(
            max_results=args.max_results,
            query_url=args.searxng_url,
        )

        # Build and run graph
        app = build_crag_graph(retriever, llm, search_tool)

        inputs = {"question": args.query}
        final_output = {}
        for output in app.stream(inputs):
            for key, value in output.items():
                logger.info(f"Node '{key}' completed")
                final_output = value

        generation = final_output.get("generation", "No answer generated.")
        root_span.set_attribute(SpanAttributes.OUTPUT_VALUE, _redact(generation[:3000]))

        # Print trace link
        phoenix_host = PHOENIX_REST_API.rstrip("/")
        if phoenix_host.endswith("/v1"):
            phoenix_host = phoenix_host[:-3]
        trace_id_hex = format(root_span.get_span_context().trace_id, "032x")
        trace_url = f"{phoenix_host}/redirects/traces/{trace_id_hex}"
        logger.info(f"🔗 Trace: {trace_url}")
        print(f"\n[Trace] {trace_url}")
        print(f"\nAnswer:\n{generation}")


if __name__ == "__main__":
    main()
