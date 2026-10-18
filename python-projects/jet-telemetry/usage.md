Here is a comprehensive guide on the philosophy and usage of `jet-telemetry`.

---

# Jet Telemetry: Usage Guide & Philosophy

## 1. Core Philosophy

`jet-telemetry` is designed to make **observability invisible** while remaining **semantically rich**. It bridges the gap between raw OpenTelemetry instrumentation and the specific needs of AI/LLM applications (RAG, Agents, Safety).

### Key Principles:

- **Decorator-First:** Use decorators (`@llm`, `@tool`, etc.) to wrap logic. This keeps business logic clean and ensures consistent attribute capture.
- **Semantic Correctness:** Automatically maps Python functions to **OpenInference Span Kinds** (e.g., `LLM`, `RETRIEVER`, `AGENT`). This ensures compatibility with Arize Phoenix and other LLM observability platforms.
- **Safety by Default:** Sensitive data (PII like passwords, API keys) is automatically redacted. Large vectors are never stored in spans; only their dimensions and metadata are captured.
- **Lazy Initialization:** The library uses a `LazyPhoenixTracer`, allowing you to apply decorators at module load time without crashing if telemetry isn't initialized yet.
- **Stackable Observability:** Decorators can be stacked. For example, you can combine `@performance_monitor`, `@cache_simulator`, and `@llm` on a single function to get latency, cache stats, and LLM semantics in one span.

---

## 2. Getting Started

### Installation

```bash
pip install jet-telemetry
```

### Initialization

Always initialize telemetry at the entry point of your application. This sets up the TracerProvider and connects to Arize Phoenix.

```python
from jet_telemetry import initialize_telemetry

# Basic initialization
initialize_telemetry(
    service_name="my-ai-app",
    endpoint="http://localhost:6006" # Default Phoenix URL
)
```

> **Note:** The `service_name` becomes the **Project Name** in Phoenix.

---

## 3. The Decorator Toolkit

The library provides specialized decorators for every layer of an AI stack.

### 🧠 Core AI Components

| Decorator | Span Kind | Use Case                                                 | Key Attributes Captured                                           |
| :-------- | :-------- | :------------------------------------------------------- | :---------------------------------------------------------------- |
| `@llm`    | `LLM`     | Wraps calls to LLM providers (OpenAI, Llama.cpp, etc.).  | `llm.model_name`, `llm.input_messages` (redacted), `llm.provider` |
| `@tool`   | `TOOL`    | Wraps external actions (API calls, DB lookups, Weather). | `tool.name`, `tool.description`, `tool.parameters`                |
| `@agent`  | `AGENT`   | Wraps the main loop of an autonomous agent.              | `agent.name`                                                      |
| `@chain`  | `CHAIN`   | Wraps orchestration logic (e.g., a full RAG pipeline).   | Acts as the root span for a trace.                                |

### 🔍 RAG & Search Components

| Decorator    | Span Kind   | Use Case                                 | Key Attributes Captured                                                           |
| :----------- | :---------- | :--------------------------------------- | :-------------------------------------------------------------------------------- |
| `@embedding` | `EMBEDDING` | Wraps text-to-vector generation.         | `embedding.model_name`, `embedding.text` (redacted), `embedding.vector_dimension` |
| `@retriever` | `RETRIEVER` | Wraps vector store or keyword search.    | `retriever.model_name`, `retrieval.query`, `retrieval.document_count`             |
| `@reranker`  | `RERANKER`  | Wraps re-scoring of retrieved documents. | `reranker.model_name`, `reranker.query`, `reranker.output_document_count`         |

### 🛡️ Safety & Evaluation

| Decorator    | Span Kind   | Use Case                           | Key Attributes Captured               |
| :----------- | :---------- | :--------------------------------- | :------------------------------------ |
| `@guardrail` | `GUARDRAIL` | Input/output safety checks.        | `guardrail.input`, `guardrail.result` |
| `@evaluator` | `EVALUATOR` | LLM-as-a-judge or scoring logic.   | N/A (Custom attributes via span)      |
| `@prompt`    | `PROMPT`    | Prompt templating/rendering steps. | N/A (Custom attributes via span)      |

---

## 4. Best Practices & Patterns

### A. The "Root Chain" Pattern

Every trace should have a root. Use `@chain` on your top-level function to group all subsequent spans under a single Trace ID.

```python
from jet_telemetry import chain, get_trace_url

@chain(name="my-rag-pipeline")
async def run_rag(query: str):
    # All nested @llm, @retriever calls will appear under this trace
    result = await retrieve_and_answer(query)
    return {"result": result, "trace_url": get_trace_url()}
```

### B. Stacking Decorators for Rich Context

You can stack multiple decorators to add cross-cutting concerns like performance monitoring or caching to a specific span.

**Example: A Tool with Retry, Cache, and Perf Monitoring**

```python
from jet_telemetry import tool, performance_monitor

# Custom decorators (defined in your app) can also be stacked
@tool(description="Search DB")
@performance_monitor(threshold_ms=200) # Adds perf.duration_ms attribute
@my_cache_decorator # Adds cache.hit attribute
async def search_db(query: str):
    # Logic here
    pass
```

_Result:_ A single `TOOL` span in Phoenix that contains tool semantics, latency metrics, and cache status.

### C. Streaming LLMs

The `@llm` decorator works seamlessly with streaming. It captures the input messages before the stream starts. Ensure you pass `messages` as a keyword argument or positional arg that the decorator can inspect.

```python
@llm(model_name="llama-3.2-3b-instruct")
async def chat(messages: list):
    stream = await client.chat.completions.create(..., stream=True)
    async for chunk in stream:
        yield chunk
```

### D. Handling Vectors

Never pass raw embedding vectors into spans. The `@embedding` decorator automatically extracts the **dimension** of the vector but discards the actual float array to keep spans small and performant.

---

## 5. Advanced Features

### Custom Attributes

If a decorator doesn't capture something specific, you can manually add attributes to the current active span using OpenTelemetry directly.

```python
from opentelemetry import trace as otel_trace

@tool(name="weather")
async def get_weather(city: str):
    span = otel_trace.get_current_span()
    if span.is_recording():
        span.set_attribute("custom.city_code", "TYO")
    # ... logic
```

### Trace Management

Use helpers to manage traces after execution.

```python
from jet_telemetry import get_trace_url, export_spans_to_jsonl

# Get a clickable link to Phoenix UI
url = get_trace_url()

# Export spans to JSONL for offline analysis or CI/CD checks
export_spans_to_jsonl(
    project_name="my-app",
    trace_id="...", # Optional, defaults to current
    output_path="trace.jsonl"
)
```

### Redaction & Hashing

- **Redaction:** Automatic for fields containing "password", "ssn", "api_key", etc.
- **Hashing:** Use `hash_prompt()` to create a unique ID for a prompt template without storing the full text.

```python
from jet_telemetry import hash_prompt, redact

prompt_hash = hash_prompt("Hello {name}") # Returns short hex string
safe_text = redact("My password is 1234") # Returns "[REDACTED...]"
```

---

## 6. Example: Full RAG Pipeline

This pattern combines retrieval, reranking, and generation into a single observable trace.

```python
from jet_telemetry import chain, embedding, retriever, reranker, llm

@embedding(model_name="nomic-embed")
def embed(text: str):
    return model.encode(text)

@retriever(name="vector-store", model_name="cosine-sim")
def search(embedding: list):
    return db.search(embedding, top_k=5)

@reranker(name="cross-encoder", model_name="bge-reranker")
def rerank(query: str, docs: list):
    return model.rerank(query, docs)

@llm(model_name="llama-3")
def generate(context: str, query: str):
    return model.chat(f"Context: {context}\nQ: {query}")

@chain(name="full-rag")
def rag_pipeline(query: str):
    emb = embed(query)
    docs = search(emb)
    ranked = rerank(query, docs)
    return generate("\n".join(ranked), query)
```

## 7. Troubleshooting

- **Empty Spans:** Ensure `initialize_telemetry()` is called **before** any decorated functions are executed.
- **Missing Attributes:** Check that you are passing arguments in a way the decorator can inspect (e.g., `messages` for `@llm`).
- **Phoenix Connection:** Verify the `endpoint` URL. If using local Phoenix, it's usually `http://localhost:6006/v1/traces`.
