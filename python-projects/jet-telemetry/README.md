# Jet Telemetry

> **Summary:** A shared, production-ready instrumentation library for Python AI applications. Provides standardized OpenTelemetry tracing for Arize Phoenix with a focus on simplicity and semantic conventions.

## 🚀 Features

- **Phoenix-Native:** Uses `arize-phoenix-otel` for automatic configuration and defaults.
- **Semantic Conventions:** Follows OpenInference standards for LLM, Tool, and Chain spans.
- **Minimalist API:** Only 4 decorators (`llm`, `tool`, `chain`, `trace`) to cover every use case.
- **Environment Driven:** Configurable via `LLM_OBS_PHOENIX_URL` for seamless local-to-prod transitions.

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
initialize_telemetry(service_name="my-rag-app")
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

| Env Variable          | Default                 | Description                                  |
| :-------------------- | :---------------------- | :------------------------------------------- |
| `LLM_OBS_PHOENIX_URL` | `http://localhost:6006` | The endpoint for the Phoenix OTLP collector. |
