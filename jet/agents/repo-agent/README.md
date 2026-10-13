# Repo Agent (Minimal)

A local, privacy-first AI agent that indexes and queries your code repositories using vector search, semantic reranking, and large language models. Built for developers who want intelligent code assistance without sending their code to the cloud.

## 🌟 Features

- **🔒 Local & Private**: All processing happens on your machine. No code leaves your environment.
- **⚡ Fast Vector Search**: Powered by LanceDB for efficient similarity search over code embeddings.
- **🎯 Semantic Reranking**: Uses cross-encoder models (with BM25 fallback) to precisely rank relevant code snippets.
- **🧠 Token-Aware Chunking**: Intelligent code splitting that respects token boundaries and maintains context overlap.
- **💬 Natural Language Queries**: Ask questions like "how to calculate sum" and get answers based on your actual codebase.
- **📊 Accurate Context Management**: Real tokenizer-based token counting prevents context window overflow.
- **🔄 Graceful Degradation**: Works even when reranker servers are unavailable via automatic BM25 fallback.

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Repository    │────▶│   Scanner    │────▶│   CodeChunker   │
│   Scanner       │     │              │     │                 │
└─────────────────┘     └──────────────┘     └────────┬────────┘
                                                       │
                                                       ▼
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│   QueryEngine   │◀────│  Retriever   │◀────│  VectorStore    │
│   (LLM +        │     │  (Search +   │     │  (LanceDB +     │
│    Context)     │     │   Rerank)    │     │   Embeddings)   │
└─────────────────┘     └──────────────┘     └─────────────────┘
```

### Components

1. **Scanner**: Identifies supported files in a repository
2. **CodeChunker**: Splits code into token-aware chunks with overlap
3. **VectorStore**: Stores and searches code embeddings using LanceDB
4. **Retriever**: Combines vector search with semantic reranking
5. **QueryEngine**: Orchestrates context assembly and LLM interaction

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Local llama.cpp servers running:
  - **Embedding Server** (e.g., `nomic-embed:1.5`)
  - **Reranker Server** (e.g., `bge-rerank:v2-m3`)
  - **LLM Server** (e.g., `qwen3.5-uncensored:2b`)

### Installation

```bash
# Clone or navigate to the repo-agent directory
cd jet_python_modules/jet/agents/repo-agent

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

#### Index a Repository

```bash
python main.py --repo /path/to/your/repository
```

#### Query the Indexed Repository

```bash
python main.py --repo /path/to/your/repository --query "how to calculate sum"
```

#### Interactive Mode

```bash
python main.py --repo /path/to/your/repository
# Then type your questions interactively
```

#### Force Re-indexing

```bash
python main.py --repo /path/to/your/repository --reindex
```

## 📋 Configuration

Edit `config.py` to customize behavior:

```python
# Server URLs (from jet.adapters.llama_cpp.config)
servers.LLM_URL      # LLM server endpoint
servers.EMBED_URL    # Embedding server endpoint
servers.RERANK_URL   # Reranker server endpoint

# Model Selection
models.LLM_MODEL     # e.g., "qwen3.5-uncensored:2b"
models.EMBED_MODEL   # e.g., "nomic-embed:1.5"
models.RERANK_MODEL  # e.g., "bge-rerank:v2-m3"

# Retrieval Settings
retrieval.TOP_K_INITIAL      # Initial candidates from vector search (default: 50)
retrieval.TOP_K_FINAL        # Final results after reranking (default: 5)
retrieval.DISTANCE_THRESHOLD # Max cosine distance (default: 0.7)

# Chunking Settings
chunking.CHUNK_SIZE    # Max tokens per chunk (default: 800)
chunking.CHUNK_OVERLAP # Overlap tokens between chunks (default: 100)
```

### Environment Variables

Set these before running if you need custom server URLs:

```bash
export LLAMA_CPP_LLM_URL="http://localhost:8080/v1"
export LLAMA_CPP_EMBED_URL="http://localhost:8081/v1"
export LLAMA_CPP_RERANK_URL="http://localhost:8082/v1"
export LLAMA_CPP_LLM_MODEL="qwen3.5-uncensored:2b"
export LLAMA_CPP_EMBED_MODEL="nomic-embed:1.5"
export LLAMA_CPP_RERANK_MODEL="bge-rerank:v2-m3"
```

## 🧪 Testing

Run the test suite:

```bash
pytest tests/ -v
```

### Test Coverage

- ✅ Scanner functionality
- ✅ Token-aware chunking with overlap
- ✅ Vector store operations (add/search)
- ✅ Retrieval pipeline with reranking
- ✅ Query engine orchestration
- ✅ Error handling and edge cases

## 📊 Performance

| Operation         | Performance            | Notes                                  |
| ----------------- | ---------------------- | -------------------------------------- |
| **Embedding**     | ~70ms for 3 chunks     | Parallel processing with deduplication |
| **Reranking**     | ~1.9s for 3 candidates | Cross-encoder with BM25 fallback       |
| **Vector Search** | <100ms                 | LanceDB cosine similarity              |
| **Total Query**   | ~2-3s                  | End-to-end retrieval + LLM generation  |

## 🔧 Technical Details

### Token-Aware Chunking

Unlike character-based approximations, Repo Agent uses real tokenizers from HuggingFace to:

- Accurately count tokens per chunk
- Respect token boundaries when splitting
- Maintain proper overlap between chunks
- Prevent context window overflow

### Embedding Pipeline

Uses `jet.adapters.llama_cpp.embed_utils` for:

- **Parallel Processing**: Thread pool for concurrent embedding requests
- **Deduplication**: Avoids redundant API calls for identical texts
- **Connection Pooling**: Reuses HTTP connections efficiently
- **Progress Tracking**: Optional Rich progress bars for large batches
- **Health Checks**: Validates server availability before processing

### Reranking Strategy

Implements a robust two-tier approach:

1. **Primary**: Cross-encoder model via llama.cpp server
2. **Fallback**: BM25 lexical search if server is unavailable
3. **Normalization**: Sigmoid score normalization for consistent ranking

### Supported File Types

```python
.py, .js, .ts, .jsx, .tsx, .java, .cpp, .c, .h, .hpp,
.go, .rs, .rb, .php, .swift, .md, .txt, .json, .yaml,
.yml, .toml
```

## 🛠️ Development

### Project Structure

```
repo-agent/
├── main.py              # CLI entry point
├── scanner.py           # Repository file scanner
├── chunker.py           # Token-aware code chunking
├── vector_store.py      # LanceDB integration
├── retriever.py         # Search + reranking pipeline
├── reranker.py          # Reranker integration
├── query_engine.py      # LLM orchestration
├── config.py            # Centralized configuration
├── utils.py             # Logging and utilities
├── requirements.txt     # Dependencies
└── tests/               # Test suite
    ├── conftest.py
    ├── test_scanner_chunker.py
    ├── test_vector_store.py
    ├── test_retrieval_pipeline.py
    └── test_query_engine.py
```

### Adding New Features

1. **New File Types**: Add extensions to `chunking.SUPPORTED_EXTS` in `config.py`
2. **Custom Chunking**: Modify `CodeChunker` class in `chunker.py`
3. **Different LLM**: Update `models.LLM_MODEL` and adjust prompt format in `query_engine.py`

### Debugging

Enable verbose logging:

```python
import logging
logging.getLogger("repo_agent").setLevel(logging.DEBUG)
```

## 🤝 Integration with Jet Modules

Repo Agent leverages production-grade components from the `jet/*` ecosystem:

- **`jet.adapters.llama_cpp.chunking_utils`**: Token-aware text chunking
- **`jet.adapters.llama_cpp.embed_utils`**: Optimized embedding pipeline
- **`jet.adapters.llama_cpp.rerank_utils`**: Robust reranking with fallback
- **`jet.adapters.llama_cpp.token_utils`**: Accurate token counting
- **`jet.adapters.llama_cpp.factory`**: Centralized client management

This ensures consistency, reliability, and performance across all Jet projects.

## ⚠️ Limitations

- **Local Servers Required**: Must have llama.cpp servers running for embedding, reranking, and LLM
- **Index Size**: Large repositories may require significant storage for embeddings
- **Language Support**: Best results with well-structured code; binary files not supported
- **Context Window**: Limited by LLM's context size (default: 16384 tokens)

## 📝 License

This project is part of the Jet Python Modules ecosystem. See the main repository for license details.

## 🙏 Acknowledgments

- **LanceDB**: For efficient vector database operations
- **llama.cpp**: For local LLM inference capabilities
- **HuggingFace Transformers**: For accurate tokenization
- **Jet Framework**: For reusable production-grade components
