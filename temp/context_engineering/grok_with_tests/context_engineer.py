import openai
import re
import os
from typing import List

openai.api_key = os.environ.get("OPENAI_API_KEY")

def summarize_document(doc: str, max_length: int = 500) -> str:
    """Summarize a document if it's too long using OpenAI."""
    if len(doc) <= max_length:
        return doc
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful summarizer. Condense the following text while preserving key information."},
            {"role": "user", "content": doc}
        ],
        max_tokens=150
    )
    return response.choices[0].message.content

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

def generate_response(query: str, docs: List[str], model: str = "gpt-4o") -> str:
    """Generate LLM response with engineered context."""
    context = engineer_context(query, docs)
    prompt = f"""
<system_instructions>
You are an AI assistant. Use the provided documents to answer the query accurately. Do not add external knowledge.
</system_instructions>

<relevant_context>
{context}
</relevant_context>

<user_query>
{query}
</user_query>
"""
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "system", "content": prompt}]
    )
    return response.choices[0].message.content