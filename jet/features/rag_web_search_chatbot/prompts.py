# Prompt templates
RAG_PROMPT = """
You are a helpful assistant. Use the following context to answer the question.
If the context is not sufficient, say so.

Context: {context}

Question: {question}

Answer:"""

ROUTER_PROMPT = """
You are an agent deciding how to answer a question.
If the question can be answered from internal knowledge (RAG), use the retriever tool.
Otherwise, use web search.

Question: {question}

Respond with: "RAG" or "WEB_SEARCH"."""
