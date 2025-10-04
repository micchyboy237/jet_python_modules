from typing import List, Dict, Any, Tuple
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

def extract_keywords_for_rag(
    docs: List[str],
    query: str,
    top_k: int = 5,
    ngram_range: Tuple[int, int] = (1, 2)
) -> List[Dict[str, Any]]:
    """
    Extract keywords from documents using KeyBERT for RAG query expansion.
    
    Args:
        docs: List of docs.
        query: Query to weight keyword extraction for relevance.
        top_k: Keywords per doc.
        ngram_range: Range for keyphrases.
    
    Returns:
        List of dicts with doc keywords and scores.
    """
    kw_model = KeyBERT()
    # Initialize embedding model for query
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = embedding_model.encode([query], convert_to_tensor=False)[0]
    
    results: List[Dict[str, Any]] = []
    for i, doc in enumerate(docs):
        # Extract keywords with query weighting
        keywords = kw_model.extract_keywords(
            doc,
            keyphrase_ngram_range=ngram_range,
            stop_words="english",
            top_n=top_k,
            use_mmr=True,
            diversity=0.5,
            doc_embeddings=query_embedding  # Weight keywords by query similarity
        )
        
        # Format as dict
        kw_list = [{"keyword": kw[0], "score": float(kw[1])} for kw in keywords]
        results.append({
            "doc_id": i,
            "doc_text": doc[:100] + "...",
            "keywords": kw_list
        })
    
    return results

# Example usage (same docs as above)
if __name__ == "__main__":
    docs: List[str] = [
        "Climate change is accelerating due to fossil fuels and deforestation.",
        "Renewable energy sources like solar and wind are essential for sustainability."
    ]
    query: str = "environmental impacts"  # Not directly used but can weight if extended

    keywords = extract_keywords_for_rag(docs, query)
    print("Keywords:", keywords)
