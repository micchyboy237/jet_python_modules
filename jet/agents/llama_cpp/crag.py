"""
CRAG Agent: Corrective Retrieval Augmented Generation with Full Observability

A dynamic RAG pipeline that retrieves documents from a local vectorstore,
evaluates their relevance using an LLM judge, and falls back to web search
when retrieval quality is insufficient. Every step is traced via OpenTelemetry
and exported to Phoenix for debugging, evaluation, and cost tracking.

Modes:
  • Full CRAG    — Provide --pdf or --index-cache to enable retrieve→evaluate→decide→generate
  • Web-only     — Omit --pdf (or use --web-only) to skip retrieval and search the web directly

Usage Examples:
  # Web-only mode (no PDF required)
  python crag.py "What are the latest AI regulations in the EU?"

  # Full CRAG with PDF encoding
  python crag.py "Explain quantum entanglement" --pdf ./physics_book.pdf

  # Reuse cached FAISS index (skips re-encoding)
  python crag.py "Summarize chapter 3" --index-cache ./cache/physics_faiss

  # Force web-only even if cache exists
  python crag.py "Breaking news today" --index-cache ./cache/physics_faiss --web-only

  # Customize retrieval and decision thresholds
  python crag.py "How does deforestation affect climate?" \\
      --pdf ./climate.pdf \\
      --retrieval-k 5 \\
      --threshold-high 0.75 \\
      --threshold-low 0.25 \\
      --chunk-size 800 \\
      --chunk-overlap 150

Environment Variables (from jet.adapters.llama_cpp.config):
  LLM_MODEL, EMBED_MODEL_LG, PHOENIX_REST_API, SEARXNG_URL, etc.

Dependencies:
  pip install langchain langchain-community faiss-cpu openinference-semantic-conventions
  pip install opentelemetry-sdk phoenix-otel pydantic python-dotenv rich
"""

import sys

sys.path.append(
    "/Users/jethroestrada/Desktop/External_Projects/AI/examples/RAG_Techniques"
)
import argparse
import json
import os
import uuid
import warnings
from typing import List, Tuple, Union

from dotenv import load_dotenv
from evaluation.evalute_rag import *
from helper_functions import replace_t_with_space
from jet.adapters.langchain.factory import get_chat_openai, get_openai_embeddings
from jet.adapters.langchain.tools.searxng_search_tool import SearXNGSearchResults
from jet.adapters.llama_cpp.config import EMBED_MODEL_LG, LLM_MODEL, PHOENIX_REST_API
from jet.logger import logger
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts.prompt import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from phoenix.otel import HTTPSpanExporter
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Suppress harmless LangChain extra_body warning
warnings.filterwarnings(
    "ignore",
    message="Parameters {'extra_body'} should be specified explicitly.*",
    category=UserWarning,
)

load_dotenv()

# --- Observability Setup (Manual Spans Only) ---
PROJECT_NAME = "crag-agent-local"
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


def encode_pdf(
    path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    cache_path: str | None = None,
) -> FAISS:
    """
    Encodes a PDF book into a vector store using OpenAI embeddings.
    Supports optional disk caching to avoid re-encoding on repeated runs.
    If cache_path exists and path is empty/unused, loads from cache directly.
    """
    # Try loading from cache first
    if cache_path and os.path.exists(cache_path):
        with tracer.start_as_current_span(
            "crag.startup.load_cached_index",
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.RETRIEVER.value,
                "crag.cache_path": cache_path,
            },
        ) as span:
            logger.info(f"Loading cached FAISS index from: {cache_path}")
            embeddings = get_openai_embeddings(EMBED_MODEL_LG)
            vectorstore = FAISS.load_local(
                cache_path, embeddings, allow_dangerous_deserialization=True
            )
            span.set_attribute("crag.cache_hit", True)
            logger.success("Cached index loaded successfully")
            return vectorstore

    # Encode from scratch
    with tracer.start_as_current_span(
        "crag.startup.encode_pdf",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.EMBEDDING.value,
            "crag.pdf_path": path,
            "crag.chunk_size": chunk_size,
            "crag.chunk_overlap": chunk_overlap,
            SpanAttributes.EMBEDDING_MODEL_NAME: EMBED_MODEL_LG,
        },
    ) as span:
        logger.info(f"Encoding PDF: {path}")
        loader = PyPDFLoader(path)
        documents = loader.load()
        span.set_attribute("crag.raw_document_count", len(documents))

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
        )
        texts = text_splitter.split_documents(documents)
        cleaned_texts = replace_t_with_space(texts)
        span.set_attribute("crag.chunk_count", len(cleaned_texts))

        embeddings = get_openai_embeddings(EMBED_MODEL_LG)
        vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

        # Save to cache if path provided
        if cache_path:
            vectorstore.save_local(cache_path)
            span.set_attribute("crag.cache_saved", cache_path)
            logger.info(f"Saved FAISS index to cache: {cache_path}")

        span.set_attribute("crag.vectorstore_type", "FAISS")
        logger.success(f"Encoded {len(cleaned_texts)} chunks into FAISS index")
        return vectorstore


# Initialize LLM and search tool once at module level (no dynamic config needed)
llm = get_chat_openai(model=LLM_MODEL, max_tokens=1000, temperature=0)
search = SearXNGSearchResults()


class RetrievalEvaluatorInput(BaseModel):
    """A relevance score between 0 and 1 indicating document query match."""

    model_config = ConfigDict(populate_by_name=True)
    relevance_score: float = Field(
        ...,
        alias="relevance",
        description="The relevance score of the document to the query. Must be a float between 0.0 and 1.0.",
    )


def retrieval_evaluator(query: str, document: str) -> float:
    """Evaluate document relevance with manual LLM span."""
    with tracer.start_as_current_span(
        "crag.llm.retrieval_evaluator",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
            SpanAttributes.LLM_MODEL_NAME: LLM_MODEL,
            SpanAttributes.INPUT_VALUE: _redact(
                f"Query: {query}\nDoc: {document[:500]}"
            ),
        },
    ) as span:
        prompt = PromptTemplate(
            input_variables=["query", "document"],
            template=(
                "On a scale from 0 to 1, how relevant is the following document to the query?\n\n"
                "Query: {query}\nDocument: {document}\n\n"
                "Respond ONLY with valid JSON in this EXACT format:\n"
                '{{"relevance_score": 0.0}}\n'
                "Do NOT include any other fields like 'reasoning'. Do NOT add explanations."
            ),
        )
        chain = prompt | llm.with_structured_output(
            RetrievalEvaluatorInput, method="json_mode"
        )
        input_variables = {"query": query, "document": document}
        result_obj = chain.invoke(input_variables)
        score = result_obj.relevance_score

        usage = _extract_token_usage(result_obj)
        _set_llm_token_attrs(span, usage)

        span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(score))
        span.set_attribute("crag.relevance_score", score)
        return score


class KnowledgeRefinementInput(BaseModel):
    """Key points extracted from a document as bullet points."""

    key_points: Union[str, List[str]] = Field(
        ..., description="The extracted key information formatted as bullet points."
    )

    @field_validator("key_points", mode="before")
    @classmethod
    def normalize_key_points(cls, v: Union[str, List[str]]) -> str:
        """Accept both str and list from LLM; always return a string."""
        if isinstance(v, list):
            return "\n".join(str(item) for item in v)
        return str(v)


def knowledge_refinement(document: str, query: str = "") -> List[str]:
    """Extract key points with manual LLM span."""
    with tracer.start_as_current_span(
        "crag.llm.knowledge_refinement",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
            SpanAttributes.LLM_MODEL_NAME: LLM_MODEL,
            SpanAttributes.INPUT_VALUE: _redact(
                f"Query: {query}\nDoc: {document[:500]}"
            ),
        },
    ) as span:
        prompt = PromptTemplate(
            input_variables=["document", "query"],
            template=(
                "Extract the key information from the following document that answers the query.\n\n"
                "Query: {query}\n"
                "Document: {document}\n\n"
                "Respond ONLY with valid JSON in this EXACT format:\n"
                '{{"key_points": "- point 1\\n- point 2"}}\n'
                "Do NOT include any other fields. Do NOT add explanations."
            ),
        )
        chain = prompt | llm.with_structured_output(
            KnowledgeRefinementInput, method="json_mode"
        )
        input_variables = {"document": document, "query": query}
        result_obj = chain.invoke(input_variables)
        result = result_obj.key_points
        points = [point.strip() for point in result.split("\n") if point.strip()]

        usage = _extract_token_usage(result_obj)
        _set_llm_token_attrs(span, usage)

        span.set_attribute(
            SpanAttributes.OUTPUT_VALUE, _redact("\n".join(points)[:2000])
        )
        span.set_attribute("crag.key_point_count", len(points))
        return points


class QueryRewriterInput(BaseModel):
    """A rewritten query optimized for web search."""

    query: str = Field(..., description="The rewritten search query.")


def rewrite_query(query: str) -> str:
    """Rewrite query with manual LLM span."""
    with tracer.start_as_current_span(
        "crag.llm.rewrite_query",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
            SpanAttributes.LLM_MODEL_NAME: LLM_MODEL,
            SpanAttributes.INPUT_VALUE: _redact(query),
        },
    ) as span:
        prompt = PromptTemplate(
            input_variables=["query"],
            template=(
                "Rewrite the following query to make it more suitable for a web search.\n\n"
                "Query: {query}\n\n"
                "Respond ONLY with valid JSON in this EXACT format:\n"
                '{{"query": "rewritten query here"}}\n'
                "Do NOT include any other fields. Do NOT add explanations."
            ),
        )
        chain = prompt | llm.with_structured_output(
            QueryRewriterInput, method="json_mode"
        )
        input_variables = {"query": query}
        result_obj = chain.invoke(input_variables)
        rewritten = result_obj.query.strip()

        usage = _extract_token_usage(result_obj)
        _set_llm_token_attrs(span, usage)

        span.set_attribute(SpanAttributes.OUTPUT_VALUE, _redact(rewritten))
        return rewritten


def parse_search_results(results_string: str) -> List[Tuple[str, str]]:
    """Parse search results into a list of title-link tuples."""
    try:
        results = json.loads(results_string)
        return [
            (result.get("title", "Untitled"), result.get("url", result.get("link", "")))
            for result in results
        ]
    except json.JSONDecodeError:
        try:
            entries = results_string.strip().split("\n\n")
            parsed = []
            for entry in entries:
                lines = entry.split("\n")
                entry_dict = {}
                for line in lines:
                    if ": " in line:
                        key, value = line.split(": ", 1)
                        entry_dict[key] = value
                title = entry_dict.get("title", "Untitled")
                url = entry_dict.get("url", "")
                parsed.append((title, url))
            return parsed if parsed else []
        except Exception as e:
            logger.warning(f"Error parsing search results text: {e}")
            return []


def retrieve_documents(query: str, faiss_index: FAISS, k: int = 3) -> List[str]:
    """Retrieve documents with manual retriever span."""
    with tracer.start_as_current_span(
        "crag.retrieve_documents",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.RETRIEVER.value,
            SpanAttributes.INPUT_VALUE: _redact(query),
            "crag.retrieval_k": k,
        },
    ) as span:
        docs = faiss_index.similarity_search(query, k=k)
        contents = [doc.page_content for doc in docs]

        span.set_attribute("crag.retrieved_doc_count", len(contents))
        for i, content in enumerate(contents):
            span.set_attribute(
                f"crag.retrieved_doc.{i}.content", _redact(content[:2000])
            )
        return contents


def evaluate_documents(query: str, documents: List[str]) -> List[float]:
    """Evaluate documents with parent CHAIN span wrapping individual evaluator calls."""
    with tracer.start_as_current_span(
        "crag.evaluate_documents",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
            "crag.doc_count_to_evaluate": len(documents),
            SpanAttributes.INPUT_VALUE: _redact(query),
        },
    ) as span:
        scores = [retrieval_evaluator(query, doc) for doc in documents]
        span.set_attribute("crag.eval_scores", json.dumps(scores))
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps(scores))
        return scores


def perform_web_search(query: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Perform web search with full manual tracing."""
    with tracer.start_as_current_span(
        "crag.web_search",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.TOOL.value,
            SpanAttributes.TOOL_NAME: "searxng_search",
            SpanAttributes.INPUT_VALUE: _redact(query),
        },
    ) as span:
        rewritten_query = rewrite_query(query)
        span.set_attribute("crag.rewritten_query", _redact(rewritten_query))

        result = search.run(rewritten_query)
        web_results = result[0] if isinstance(result, tuple) else result
        span.set_attribute("crag.raw_search_result_length", len(str(web_results)))

        web_knowledge_list = knowledge_refinement(web_results, rewritten_query)
        sources = parse_search_results(web_results)
        web_knowledge = (
            "\n".join(web_knowledge_list)
            if isinstance(web_knowledge_list, list)
            else web_knowledge_list
        )

        span.set_attribute(SpanAttributes.OUTPUT_VALUE, _redact(web_knowledge[:2000]))
        span.set_attribute("crag.web_source_count", len(sources))
        return web_knowledge, sources


def generate_response(
    query: str, knowledge: str, sources: List[Tuple[str, str]]
) -> str:
    """Generate final response with manual LLM span."""
    with tracer.start_as_current_span(
        "crag.llm.generate_response",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
            SpanAttributes.LLM_MODEL_NAME: LLM_MODEL,
            SpanAttributes.INPUT_VALUE: _redact(
                f"Query: {query}\nKnowledge length: {len(knowledge)}"
            ),
            "crag.source_count": len(sources),
        },
    ) as span:
        response_prompt = PromptTemplate(
            input_variables=["query", "knowledge", "sources"],
            template=(
                "Based on the following knowledge, answer the query. "
                "Include the sources with their links (if available) at the end of your answer:\n"
                "Query: {query}\nKnowledge: {knowledge}\nSources: {sources}\nAnswer:"
            ),
        )
        input_variables = {
            "query": query,
            "knowledge": knowledge,
            "sources": "\n".join(
                [f"{title}: {link}" if link else title for title, link in sources]
            ),
        }
        response_chain = response_prompt | llm
        response_msg = response_chain.invoke(input_variables)
        response_content = response_msg.content

        usage = _extract_token_usage(response_msg)
        _set_llm_token_attrs(span, usage)

        span.set_attribute(
            SpanAttributes.OUTPUT_VALUE, _redact(response_content[:3000])
        )
        return response_content


def crag_process(
    query: str,
    faiss_index: FAISS | None = None,
    retrieval_k: int = 3,
    threshold_high: float = 0.7,
    threshold_low: float = 0.3,
) -> str:
    """
    Process a query with optional vectorstore. If faiss_index is None, runs in web-only mode.
    Fully observable with manual OpenTelemetry spans. All thresholds and retrieval params are dynamic.
    """
    session_id = str(uuid.uuid4())
    logger.info(f"Processing query: {query}")
    web_only_mode = faiss_index is None

    with tracer.start_as_current_span(
        "crag.session",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.AGENT.value,
            SpanAttributes.SESSION_ID: session_id,
            SpanAttributes.INPUT_VALUE: _redact(query),
            "crag.query": _redact(query),
            "crag.retrieval_k": retrieval_k,
            "crag.threshold_high": threshold_high,
            "crag.threshold_low": threshold_low,
            "crag.web_only_mode": web_only_mode,
        },
    ) as root_span:
        sources = []
        decision = ""
        final_knowledge = ""

        if web_only_mode:
            # --- WEB-ONLY PATH ---
            decision = "web_only"
            logger.info("Action: Web-only mode (no vectorstore provided)")

            with tracer.start_as_current_span(
                "crag.decide",
                attributes={
                    SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
                    "crag.decision": decision,
                    "crag.mode": "web_only",
                },
            ) as decide_span:
                final_knowledge, sources = perform_web_search(query)
                if isinstance(final_knowledge, list):
                    final_knowledge = "\n".join(final_knowledge)
        else:
            # --- FULL CRAG PATH ---
            # Retrieve
            retrieved_docs = retrieve_documents(query, faiss_index, k=retrieval_k)
            logger.info(f"Retrieved {len(retrieved_docs)} documents")

            # Evaluate
            eval_scores = evaluate_documents(query, retrieved_docs)
            logger.info(f"Evaluation scores: {eval_scores}")

            # Decide
            max_score = max(eval_scores) if eval_scores else 0.0

            with tracer.start_as_current_span(
                "crag.decide",
                attributes={
                    SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
                    "crag.max_relevance_score": max_score,
                    "crag.threshold_high": threshold_high,
                    "crag.threshold_low": threshold_low,
                },
            ) as decide_span:
                if max_score > threshold_high:
                    decision = "correct"
                    logger.info(
                        f"Action: Correct (score={max_score:.2f} > {threshold_high}) - Using retrieved document"
                    )
                    best_doc = retrieved_docs[eval_scores.index(max_score)]
                    final_knowledge = best_doc
                    sources.append(("Retrieved document", ""))
                elif max_score < threshold_low:
                    decision = "incorrect"
                    logger.info(
                        f"Action: Incorrect (score={max_score:.2f} < {threshold_low}) - Performing web search"
                    )
                    final_knowledge, sources = perform_web_search(query)
                    if isinstance(final_knowledge, list):
                        final_knowledge = "\n".join(final_knowledge)
                else:
                    decision = "ambiguous"
                    logger.info(
                        f"Action: Ambiguous ({threshold_low} <= score={max_score:.2f} <= {threshold_high}) - Combining sources"
                    )
                    best_doc = retrieved_docs[eval_scores.index(max_score)]
                    retrieved_knowledge_list = knowledge_refinement(best_doc, query)
                    web_knowledge, web_sources = perform_web_search(query)
                    retrieved_knowledge = (
                        "\n".join(retrieved_knowledge_list)
                        if isinstance(retrieved_knowledge_list, list)
                        else retrieved_knowledge_list
                    )
                    web_knowledge = (
                        "\n".join(web_knowledge)
                        if isinstance(web_knowledge, list)
                        else web_knowledge
                    )
                    final_knowledge = "\n".join([retrieved_knowledge, web_knowledge])
                    sources = [("Retrieved document", "")] + web_sources

                decide_span.set_attribute("crag.decision", decision)

        if isinstance(final_knowledge, list):
            final_knowledge = "\n".join(final_knowledge)

        logger.info("Final knowledge:")
        logger.info(final_knowledge)
        logger.info("Sources:")
        for title, link in sources:
            logger.info(f"{title}: {link}" if link else title)

        # Generate
        logger.info("Generating response...")
        response = generate_response(query, final_knowledge, sources)
        logger.success("Response generated")

        # Finalize root span
        root_span.set_attribute(SpanAttributes.OUTPUT_VALUE, _redact(response[:3000]))
        root_span.set_attribute("crag.decision_path", decision)
        root_span.set_attribute("crag.final_source_count", len(sources))

        # Print trace link
        phoenix_host = PHOENIX_REST_API.rstrip("/")
        if phoenix_host.endswith("/v1"):
            phoenix_host = phoenix_host[:-3]
        trace_id_hex = format(root_span.get_span_context().trace_id, "032x")
        trace_url = f"{phoenix_host}/redirects/traces/{trace_id_hex}"
        logger.info(f"🔗 Trace: {trace_url}")
        print(f"[Trace] {trace_url}")

        return response


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser with sensible defaults."""
    parser = argparse.ArgumentParser(
        description="CRAG Agent: Corrective RAG with full observability. Supports web-only mode when --pdf is omitted.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="What are the latest developments in renewable energy?",
        help="User query to process",
    )
    parser.add_argument(
        "-p",
        "--pdf",
        type=str,
        default=None,
        help="Path to PDF file for encoding. Omit to run in web-only mode.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Text splitter chunk size (ignored in web-only mode)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Text splitter chunk overlap (ignored in web-only mode)",
    )
    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=3,
        help="Number of documents to retrieve from FAISS (ignored in web-only mode)",
    )
    parser.add_argument(
        "--threshold-high",
        type=float,
        default=0.7,
        help="Relevance score threshold for 'Correct' decision (ignored in web-only mode)",
    )
    parser.add_argument(
        "--threshold-low",
        type=float,
        default=0.3,
        help="Relevance score threshold for 'Incorrect' decision (ignored in web-only mode)",
    )
    parser.add_argument(
        "--index-cache",
        type=str,
        default=None,
        help="Path to save/load FAISS index cache. If --pdf omitted but cache exists, loads it.",
    )
    parser.add_argument(
        "--web-only",
        action="store_true",
        help="Force web-only mode even if --pdf or --index-cache is provided",
    )
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    vectorstore = None

    # Determine execution mode
    if args.web_only:
        logger.info("Web-only mode forced via --web-only flag")
    elif args.pdf:
        # Validate PDF path exists
        if not os.path.isfile(args.pdf):
            logger.error(f"PDF not found: {args.pdf}")
            sys.exit(1)
        vectorstore = encode_pdf(
            path=args.pdf,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            cache_path=args.index_cache,
        )
    elif args.index_cache and os.path.exists(args.index_cache):
        # Try loading from cache without PDF
        logger.info(
            f"No --pdf provided; attempting to load cached index: {args.index_cache}"
        )
        vectorstore = encode_pdf(
            path="",  # Unused when cache hits
            cache_path=args.index_cache,
        )
    else:
        logger.info(
            "No --pdf or valid --index-cache provided → running in web-only mode"
        )

    # Run CRAG pipeline
    result = crag_process(
        query=args.query,
        faiss_index=vectorstore,  # May be None
        retrieval_k=args.retrieval_k,
        threshold_high=args.threshold_high,
        threshold_low=args.threshold_low,
    )

    print(f"\n{'=' * 60}")
    print(f"Mode: {'Web-only' if vectorstore is None else 'Full CRAG'}")
    print(f"Query: {args.query}")
    print(f"Answer: {result}")
    print(f"{'=' * 60}")
