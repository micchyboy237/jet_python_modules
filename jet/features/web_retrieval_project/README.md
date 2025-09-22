# Web Retrieval Project

Retrieve relevant content from recursively scraped URLs using LangChain.

## Setup

1. Install Python 3.9+.
2. Create a virtual environment: `python -m venv venv`
3. Activate: `source venv/bin/activate` (Mac) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Set `OPENAI_API_KEY` environment variable for embeddings/LLM.
6. Run tests: `pytest tests/`

## Usage

```python
from src.scraper import scrape_recursive_url
from src.indexer import index_scraped_docs
from src.retriever import rag_query, RAGInput

docs = scrape_recursive_url("https://docs.python.org/3.9/", max_depth=1)
vectorstore = index_scraped_docs(docs)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
answer = rag_query({"query": "What is Python's garbage collector?", "retriever": retriever})
```
