# Repo Agent

A local, privacy-first **AI Agent** that indexes, explores, and queries your code repositories. Built for developers who want intelligent code assistance without sending their code to the cloud.

Unlike simple RAG systems, Repo Agent uses an **Agentic Workflow** to autonomously plan, search, and retrieve information using a suite of specialized tools.

## 🚀 Features

- **🤖 Agentic Workflow**: Uses a ReAct (Reason + Act) loop to autonomously decide when to search, read files, or grep for symbols.
- **🛠️ Tool Use**: Equipped with `retrieve_context`, `read_file`, and `grep_search` tools for precise code exploration.
- **👁️ Full Observability**: Integrated with **Jet Telemetry** and **Arize Phoenix** to trace every LLM call, tool execution, and retrieval step.
- **🏠 Local & Private**: All processing happens on your machine using local llama.cpp servers. No code leaves your environment.
- **⚡ Hybrid Retrieval**: Combines semantic vector search (LanceDB) with lexical reranking (BGE/BM25) for high-precision results.
- **🧩 Token-Aware Chunking**: Uses production-grade `jet.adapters.llama_cpp.chunking_utils` to respect code structure and token boundaries.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Query / Task                       │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                  CodeAgent (agent.py)                       │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              ReAct Planning Loop                      │  │
│  │  1. Plan Next Step (LLM)                              │  │
│  │  2. Select Tool (Retrieve, Read, Grep)                │  │
│  │  3. Execute Tool                                      │  │
│  │  4. Observe Result                                    │  │
│  └───────────────────────────────────────────────────────┘  │
└───────────┬──────────────────────┬──────────────────────────┘
            │                      │
            ▼                      ▼
┌──────────────────────┐  ┌──────────────────────────┐
│   Retriever          │  │   RepoTools              │
│  (retriever.py)      │  │  (tools.py)              │
│                      │  │                          │
│  1. Vector Search    │  │  - read_file()           │
│  2. Rerank (BGE)     │  │  - grep_search()         │
└──────────┬───────────┘  └──────────────────────────┘
           │
           ▼
┌──────────────────────┐
│   VectorStore        │
│  (vector_store.py)   │
│                      │
│  - LanceDB Index     │
│  - Embeddings (Nomic)│
└──────────────────────┘
```

### Components

1.  **`agent.py`**: The core orchestrator. Manages the agentic loop using `jet.adapters.llama_cpp.llm_utils_observed.achat`.
2.  **`tools.py`**: Defines the capabilities (`@tool`) the agent can use to interact with the filesystem.
3.  **`scanner.py` & `chunker.py`**: Handles repository scanning and token-aware chunking using `jet.adapters.llama_cpp.chunking_utils`.
4.  **`vector_store.py`**: Manages LanceDB indexing and semantic search using `jet.adapters.llama_cpp.embed_utils`.
5.  **`retriever.py` & `reranker.py`**: Implements hybrid retrieval with cross-encoder reranking via `jet.adapters.llama_cpp.rerank_utils`.

## 📦 Quick Start

### Prerequisites

- Python 3.10+
- Local llama.cpp servers running:
  - **LLM Server**: e.g., `qwen3.5-uncensored:2b` (Port 8080)
  - **Embedding Server**: e.g., `nomic-embed:1.5` (Port 8081)
  - **Reranker Server**: e.g., `bge-rerank:v2-m3` (Port 8082)
- **Phoenix Server**: Running locally for observability (Port 6006).

### Installation

```bash
cd jet_python_modules/jet/agents/repo-agent
pip install -r requirements.txt
```

### Basic Usage

#### 1. Index a Repository

```bash
python main.py --repo /path/to/your/repository
```

#### 2. Query the Agent (Interactive Mode)

```bash
python main.py --repo /path/to/your/repository
```

_Example Interaction:_

> **You:** How does the authentication module work?
> **Agent:** I'll start by retrieving context about authentication... [Uses `retrieve_context`]
> **Agent:** I see references to `auth_service.py`. Let me read the full file to understand the flow... [Uses `read_file`]
> **Agent:** Based on `auth_service.py`, the module uses JWT tokens validated against...

#### 3. Single Query Mode

```bash
python main.py --repo /path/to/your/repository --query "Find all functions that handle database connections"
```

## 🔍 Observability

Repo Agent is fully instrumented using **Jet Telemetry**. Every action is traced in Arize Phoenix:

- **Agent Spans**: See the full reasoning loop and decision-making process.
- **Tool Spans**: Monitor exactly which files were read or searched.
- **LLM Spans**: Track token usage, latency, and input/output messages.
- **Retrieval Spans**: Visualize vector search and reranking scores.

To view traces, open your local Phoenix UI at `http://localhost:6006`.

## ⚙️ Configuration

Edit `config.py` to customize behavior:

```python
# Model Selection (from jet.adapters.llama_cpp.config)
models.LLM_MODEL     # e.g., "qwen3.5-uncensored:2b"
models.EMBED_MODEL   # e.g., "nomic-embed:1.5"
models.RERANK_MODEL  # e.g., "bge-rerank:v2-m3"

# Retrieval Settings
retrieval.TOP_K_INITIAL      # Initial candidates from vector search (default: 50)
retrieval.TOP_K_FINAL        # Final results after reranking (default: 5)

# Chunking Settings
chunking.CHUNK_SIZE    # Max tokens per chunk (default: 800)
chunking.CHUNK_OVERLAP # Overlap tokens between chunks (default: 100)
```

## 🧪 Testing

Run the integration test suite:

```bash
pytest tests/ -v
```

## 📚 Technical Details

### Agentic Loop

The agent uses `achat` from `llm_utils_observed.py`, which automatically handles:

- Parsing tool calls from the LLM.
- Executing the corresponding Python function.
- Feeding the result back into the context window.
- Repeating until a final answer is reached or `max_tool_rounds` is hit.

### Token-Aware Chunking

Uses `chunk_texts_with_data` from `jet.adapters.llama_cpp.chunking_utils` to ensure chunks are split at semantic boundaries while respecting exact token limits of the embedding model.

### Supported File Types

`.py, .js, .ts, .jsx, .tsx, .java, .cpp, .c, .h, .hpp, .go, .rs, .rb, .php, .swift, .md, .txt, .json, .yaml, .yml, .toml`

## 🤝 Integration with Jet Modules

Repo Agent leverages production-grade components from the `jet/*` ecosystem:

- **`jet.adapters.llama_cpp.llm_utils_observed`**: For agentic tool loops and observability.
- **`jet_telemetry`**: For standardized OpenInference tracing.
- **`jet.adapters.llama_cpp.chunking_utils`**: For robust text splitting.
- **`jet.adapters.llama_cpp.embed_utils`**: For parallelized embedding generation.
- **`jet.adapters.llama_cpp.rerank_utils`**: For precision reranking with BM25 fallback.
