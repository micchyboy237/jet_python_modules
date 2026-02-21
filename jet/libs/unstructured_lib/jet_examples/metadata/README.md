# Local Metadata-Aware Document Retriever (Unstructured + ChromaDB)

**Fully local**, offline RAG-style retriever using:

- `unstructured` → intelligent document partitioning (PDF, DOCX, HTML, images, etc.)
- Rich element metadata preservation
- `sentence-transformers` embeddings
- ChromaDB (persistent, local vector store with metadata filtering)

Supports filtering by `element_type` (NarrativeText, Table, ListItem, Title, …), `source_file`, custom metadata, etc.

## Features

- Ingest single files or entire directories
- Persistent local Chroma collection
- Semantic search + metadata filtering
- Beautiful console output with `rich`
- Progress bars with `tqdm`

## Installation

```bash
# Recommended: in a virtual environment
pip install \
  unstructured \
  unstructured[all-docs] \     # or only the formats you need
  chromadb \
  sentence-transformers \
  rich \
  tqdm
```

## Usage Examples

```bash
# 1. Ingest a single file
python main.py ingest --file report.pdf

# 2. Ingest everything in a folder (very common)
python main.py ingest --dir ./documents --pattern "**/*.pdf"

# 3. Search with no filter
python main.py query "What is the revenue in 2024?" --top-k 6

# 4. Search only tables
python main.py query "profit margin" --filter-element-type Table --top-k 4

# 5. Search only from specific document
python main.py query "ESG goals" --filter-source-file sustainability-2025.pdf

# 6. Combine filters (JSON syntax)
python main.py query "AI strategy" \
  --filter '{"element_type":"NarrativeText","source_file":"strategy.pdf"}' \
  --top-k 3

# 7. Show help
python main.py --help
python main.py ingest --help
python main.py query --help
```

## Default Settings (configurable via code)

```python
collection_name      = "local_docs_v1"
persist_directory    = "./chroma_db"
embedding_model_name = "all-MiniLM-L6-v2"
default_top_k        = 5
```

You can change these defaults directly in `main.py`.

## Recommended Embedding Models

| Model                                 | Speed | Quality | Size    | Language     |
| ------------------------------------- | ----- | ------- | ------- | ------------ |
| all-MiniLM-L6-v2 (default)            | ★★★★★ | ★★★☆☆   | ~80 MB  | English+     |
| all-mpnet-base-v2                     | ★★★☆☆ | ★★★★☆   | ~400 MB | English+     |
| paraphrase-multilingual-mpnet-base-v2 | ★★☆☆☆ | ★★★★☆   | ~700 MB | Multilingual |
| BAAI/bge-small-en-v1.5                | ★★★★☆ | ★★★★☆   | ~130 MB | English      |

Change via `--embedding-model` or directly in config.
