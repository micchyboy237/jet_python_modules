import sys

sys.path.append(
    "/Users/jethroestrada/Desktop/External_Projects/AI/examples/RAG_Techniques"
)
import json
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
        # LangChain AIMessage stores usage in response_metadata or usage_metadata
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


def encode_pdf(path, chunk_size=1000, chunk_overlap=200):
    """
    Encodes a PDF book into a vector store using OpenAI embeddings.
    Wrapped in a startup span for observability.
    """
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

        span.set_attribute("crag.vectorstore_type", "FAISS")
        logger.success(f"Encoded {len(cleaned_texts)} chunks into FAISS index")
        return vectorstore


path = "/Users/jethroestrada/Desktop/External_Projects/AI/examples/RAG_Techniques/data/Understanding_Climate_Change.pdf"
vectorstore = encode_pdf(path)

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

        # Extract token usage from the underlying AIMessage if accessible
        # Note: with_structured_output may strip metadata; fallback to 0 if unavailable
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
    """
    Parse search results into a list of title-link tuples.
    Handles both JSON and text-based formats.
    """
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


def crag_process(query: str, faiss_index: FAISS) -> str:
    """
    Process a query by retrieving, evaluating, and using documents or performing a web search to generate a response.
    Fully observable with manual OpenTelemetry spans.
    """
    session_id = str(uuid.uuid4())
    logger.info(f"Processing query: {query}")

    with tracer.start_as_current_span(
        "crag.session",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.AGENT.value,
            SpanAttributes.SESSION_ID: session_id,
            SpanAttributes.INPUT_VALUE: _redact(query),
            "crag.query": _redact(query),
        },
    ) as root_span:
        # Retrieve
        retrieved_docs = retrieve_documents(query, faiss_index)
        logger.info(f"Retrieved {len(retrieved_docs)} documents")

        # Evaluate
        eval_scores = evaluate_documents(query, retrieved_docs)
        logger.info(f"Evaluation scores: {eval_scores}")

        # Decide
        max_score = max(eval_scores) if eval_scores else 0.0
        sources = []
        decision = ""

        with tracer.start_as_current_span(
            "crag.decide",
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
                "crag.max_relevance_score": max_score,
                "crag.threshold_high": 0.7,
                "crag.threshold_low": 0.3,
            },
        ) as decide_span:
            if max_score > 0.7:
                decision = "correct"
                logger.info("Action: Correct - Using retrieved document")
                best_doc = retrieved_docs[eval_scores.index(max_score)]
                final_knowledge = best_doc
                sources.append(("Retrieved document", ""))
            elif max_score < 0.3:
                decision = "incorrect"
                logger.info("Action: Incorrect - Performing web search")
                final_knowledge, sources = perform_web_search(query)
                if isinstance(final_knowledge, list):
                    final_knowledge = "\n".join(final_knowledge)
            else:
                decision = "ambiguous"
                logger.info(
                    "Action: Ambiguous - Combining retrieved document and web search"
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


if __name__ == "__main__":
    query = "What are the main causes of climate change?"
    result = crag_process(query, vectorstore)
    print(f"Query: {query}")
    print(f"Answer: {result}")
