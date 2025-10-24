import re
from typing import List

from jet.adapters.llama_cpp.llm import LlamacppLLM

def summarize_document(doc: str, max_length: int = 500) -> str:
    if len(doc) <= max_length:
        return doc
    llm = LlamacppLLM(model="qwen3-instruct-2507:4b", base_url="http://shawn-pc.local:8080/v1", verbose=True)
    messages = [
        {
            "role": "system",
            "content": "You are a helpful summarizer who condenses text while preserving all key information."
        },
        {
            "role": "user",
            "content": f"{doc}"
        }
    ]
    return llm.chat(messages, temperature=0.0, max_tokens=150)

def rank_documents(query: str, docs: List[str]) -> List[str]:
    """Rank documents by simple keyword relevance (count matching words)."""
    query_words = set(re.findall(r'\w+', query.lower()))
    ranked = sorted(
        docs,
        key=lambda d: len(query_words.intersection(re.findall(r'\w+', d.lower()))),
        reverse=True
    )
    return ranked[:3]  # Select top 3 to fit context window

def engineer_context(query: str, docs: List[str]) -> str:
    """Engineer the context: summarize, rank, and format."""
    summarized_docs = [summarize_document(doc) for doc in docs]
    relevant_docs = rank_documents(query, summarized_docs)
    context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(relevant_docs)])
    return context

def generate_response(query: str, docs: List[str], model: str = "qwen3-instruct-2507:4b") -> str:
    context = engineer_context(query, docs)
    llm = LlamacppLLM(model=model, base_url="http://shawn-pc.local:8080/v1")
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant. Use the provided documents to answer the query accurately. "
                "Do not add external knowledge."
            )
        },
        {
            "role": "user",
            "content": f"<relevant_context>\n{context}\n</relevant_context>\n<user_query>\n{query}\n</user_query>"
        }
    ]
    return llm.chat(messages, temperature=0.0)
