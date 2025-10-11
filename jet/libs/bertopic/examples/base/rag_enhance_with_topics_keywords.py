from typing import List, Dict, Any
from .extract_topics_for_rag import extract_topics_for_rag
from .extract_keywords_for_rag import extract_keywords_for_rag

def rag_enhance_with_topics_keywords(
    docs: List[str],
    query: str
) -> Dict[str, Any]:
    """Hybrid function for RAG metadata generation."""
    topics, summaries = extract_topics_for_rag(docs, query)
    keywords = extract_keywords_for_rag(docs, query)
    
    return {
        "query": query,
        "topics": topics,
        "summaries": summaries,
        "doc_keywords": keywords
    }

# Example usage
if __name__ == "__main__":
    docs: List[str] = [
        "Climate change is accelerating due to fossil fuels and deforestation.",
        "Renewable energy sources like solar and wind are essential for sustainability.",
        "Global warming impacts include rising sea levels and extreme weather."
    ]
    query: str = "environmental impacts"

    enhanced = rag_enhance_with_topics_keywords(docs, query)
    print("Enhanced RAG Metadata:", enhanced)
