# Jet Telemetry

> **Summary:** A shared, production-ready instrumentation library for Python applications. Provides standardized OpenTelemetry tracing for Arize Phoenix with specialized support for LLMs, RAG pipelines, Agents, and Safety/Evaluation workflows.

## 🚀 Features

- **Phoenix-Native:** Uses `arize-phoenix-otel` for automatic configuration and defaults.
- **Comprehensive AI Semantics:** Specialized decorators for every layer of an AI stack:
  - **Core:** `@llm`, `@tool`, `@chain`, `@agent`
  - **RAG & Search:** `@embedding`, `@retriever`, `@reranker`
  - **Safety & Eval:** `@guardrail`, `@evaluator`, `@prompt`
- **Smart Attribute Capture:** Automatically captures model names, input messages, and retrieval metadata while redacting sensitive data (PII) and excluding large vectors.
- **Minimalist API:** Easy-to-use decorators that support both synchronous and asynchronous workflows.
- **Flexible Configuration:** Supports environment variables or direct code overrides for the collector endpoint.
- **Trace Management:** Built-in helpers to generate shareable trace URLs and export spans to JSONL for offline analysis.

## 🛠️ Installation

Add this package to your project's `pyproject.toml` or install from your internal registry:

```bash
pip install jet-telemetry
```
