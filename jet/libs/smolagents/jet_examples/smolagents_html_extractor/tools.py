from typing import List, Dict, Any
from bs4 import BeautifulSoup
import json
from pathlib import Path

def chunk_html(
    html_content: str,
    window_size: int = 4000,
    overlap: int = 800
) -> List[Dict[str, Any]]:
    """Split HTML into overlapping text chunks"""
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text(separator=' ', strip=True)
    
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + window_size, len(text))
        chunk_text = text[start:end]
        
        # Try to end at a space to avoid cutting words
        if end < len(text):
            last_space = chunk_text.rfind(' ')
            if last_space > 0:
                end = start + last_space
                chunk_text = text[start:end]
        
        chunks.append({
            "index": len(chunks),
            "start_char": start,
            "end_char": end,
            "text": chunk_text
        })
        
        start = end - overlap
        if start >= len(text):
            break
            
    return chunks


def extract_relevant_content(
    chunk_text: str,
    query: str,
    max_items_per_chunk: int = 5
) -> List[Dict[str, Any]]:
    """
    Dummy extraction function - in real use this would call an LLM
    Here we just do simple keyword matching for demonstration
    """
    query_words = set(query.lower().split())
    sentences = chunk_text.split('. ')
    
    results = []
    for i, sentence in enumerate(sentences):
        if len(results) >= max_items_per_chunk:
            break
        sentence_lower = sentence.lower()
        score = sum(1 for word in query_words if word in sentence_lower)
        if score >= 1:  # at least one matching word
            results.append({
                "chunk_index": None,  # filled later
                "sentence_index": i,
                "text": sentence.strip() + ".",
                "score": score,
                "relevance": "high" if score >= 3 else "medium"
            })
    
    return sorted(results, key=lambda x: x["score"], reverse=True)


def format_final_results(results: List[Dict]) -> str:
    """Format all collected results into readable output"""
    if not results:
        return "No relevant content found."
        
    lines = ["Found relevant content:", ""]
    for i, item in enumerate(results, 1):
        lines.append(f"{i}. [{item['relevance'].upper()}] Score: {item['score']}")
        lines.append(f"   {item['text']}")
        lines.append("")
    return "\n".join(lines)
