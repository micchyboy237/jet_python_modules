# Context Engineering — chatgpt_no_tests

Small, local utilities for retrieval + context engineering used in examples and tests.

Contents
- [context_engineer.py](context_engineering/chatgpt_no_tests/context_engineer.py) — core utilities and pipeline:
  - [`context_engineer.Embedder`](context_engineering/chatgpt_no_tests/context_engineer.py)
  - [`context_engineer.retrieve_top_k`](context_engineering/chatgpt_no_tests/context_engineer.py)
  - [`context_engineer.answer_with_context_engineering`](context_engineering/chatgpt_no_tests/context_engineer.py)
- [test_context_engineer.py](context_engineering/chatgpt_no_tests/test_context_engineer.py) — a small pytest that validates retrieval behavior.

Quick overview
- `Embedder` is a thin wrapper around a sentence-transformers model that returns L2-normalized embeddings.
- `retrieve_top_k` chunks documents, encodes them, and returns the top-k most similar chunks for a query.
- `answer_with_context_engineering` runs a full pipeline (retrieve → optional LLM condense → prompt assembly → LLM) and requires an OpenAI API key for the LLM steps.

Requirements
- Python 3.8+
- Install core packages used in this folder:
  - sentence-transformers
  - numpy
  - pytest (for running tests)
  - openai (only required if you call functions that use the LLM; tests in this folder do not call OpenAI)

Installation
```sh
pip install sentence-transformers numpy pytest openai