# Jet Telemetry

> **Summary:** A shared, production-ready instrumentation library for Python applications. Provides standardized OpenTelemetry tracing for Arize Phoenix with a focus on simplicity and semantic conventions.

## 🚀 Features

- **Phoenix-Native:** Uses `arize-phoenix-otel` for automatic configuration and defaults.
- **Semantic Conventions:** Follows OpenInference standards for LLM, Tool, and Chain spans.
- **Minimalist API:** Only 4 decorators (`llm`, `tool`, `chain`, `trace`) to cover every use case.
- **Flexible Configuration:** Supports environment variables or direct code overrides for the collector endpoint.

## 🛠️ Installation

Add this package to your project's `pyproject.toml` or install from your internal registry:

```bash
pip install jet-telemetry
```

## 💡 Usage

### 1. Initialize

Call this once at the start of your application:

```python
from jet_telemetry import initialize_telemetry

# Option A: Use environment variable (PHOENIX_ENDPOINT)
initialize_telemetry(service_name="my-rag-app")

# Option B: Override endpoint directly in code
initialize_telemetry(service_name="my-rag-app", endpoint="http://prod-phoenix:6006")
```

### 2. Instrument Code

Use the decorators to categorize your functions:

```python
from jet_telemetry import llm, tool, chain

@tool
async def search_db(query): ...

@llm(model_name="gpt-4o")
async def generate_answer(context): ...

@chain
async def rag_pipeline(question): ...
```

## ⚙️ Configuration

| Env Variable       | Default                 | Description                                  |
| :----------------- | :---------------------- | :------------------------------------------- |
| `PHOENIX_ENDPOINT` | `http://localhost:6006` | The endpoint for the Phoenix OTLP collector. |
