from typing import List, TypedDict
from jet.adapters.bertopic import BERTopic
from jet.file.utils import load_file, save_file
from jet.logger import logger
import os
import shutil

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(
        os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

class Topic(TypedDict):
    rank: int
    doc_index: int
    score: float
    text: str
    
def search_topics(
    query: str,
    documents: List[str],
    model: str = "embeddinggemma",
    top_k: int = None
) -> List[Topic]:
    """Extract topics from documents using BERTopic.
    
    Args:
        query: Search query to find relevant topics
        documents: List of documents to analyze
        model: Embedding model to use
        top_k: Number of top topics to return (if None, return all)
        
    Returns:
        List of Topic objects with rank, doc_index, score, and text
    """
    if not documents:
        return []
    
    try:
        logger.info(f"Starting topic extraction for {len(documents)} documents")
        
        # Create BERTopic model with custom embedding model
        # Use the specified model for embeddings
        # embedding_model = SentenceTransformer(model)
        topic_model = BERTopic(
            # embedding_model=embedding_model,
            calculate_probabilities=True,
            verbose=True,
        )
        
        # Fit the model to documents
        logger.info("Fitting BERTopic model...")
        topics, probs = topic_model.fit_transform(documents)
        
        # Get topic information
        topic_info = topic_model.get_topic_info()
        logger.info(f"Found {len(topic_info)} topics")
        
        # Find topics similar to the query
        logger.info(f"Finding topics similar to query: '{query}'")
        similar_topics, similarities = topic_model.find_topics(query, top_n=len(topic_info))
        
        # Create topic results
        results = []
        for rank, (topic_id, similarity) in enumerate(zip(similar_topics, similarities)):
            if topic_id == -1:  # Skip outlier topic
                continue
                
            # Get topic words
            topic_words = topic_model.get_topic(topic_id)
            if not topic_words:
                continue
                
            # Create topic text from top words
            topic_text = " ".join([word[0] for word in topic_words[:5]])
            
            # Find the best document for this topic
            topic_docs = [i for i, t in enumerate(topics) if t == topic_id]
            if not topic_docs:
                continue
                
            # Get the document with highest probability for this topic
            best_doc_idx = None
            best_score = 0.0
            for doc_idx in topic_docs:
                if probs is not None and doc_idx < len(probs):
                    doc_probs = probs[doc_idx]
                    if topic_id < len(doc_probs):
                        topic_prob = doc_probs[topic_id]
                        if topic_prob > best_score:
                            best_score = topic_prob
                            best_doc_idx = doc_idx
            
            if best_doc_idx is not None:
                results.append({
                    "rank": rank + 1,
                    "doc_index": best_doc_idx,
                    "score": float(similarity),
                    "text": topic_text
                })
        
        # Sort by similarity score (descending)
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Apply top_k filter if specified
        if top_k is not None:
            results = results[:top_k]
            
        logger.info(f"Returning {len(results)} topics")
        return results
        
    except ImportError as e:
        logger.error(f"BERTopic not available: {e}")
        # Fallback to simple keyword-based topic extraction
        return _fallback_topic_extraction(query, documents, top_k)
    except Exception as e:
        logger.error(f"Error in topic extraction: {e}")
        return _fallback_topic_extraction(query, documents, top_k)

def _fallback_topic_extraction(
    query: str, 
    documents: List[str], 
    top_k: int = None
) -> List[Topic]:
    """Fallback topic extraction using simple keyword matching."""
    import re
    from collections import Counter
    
    # Extract query keywords
    query_words = set(re.findall(r'\b\w+\b', query.lower()))
    
    results = []
    for doc_idx, doc in enumerate(documents):
        # Extract document words
        doc_words = re.findall(r'\b\w+\b', doc.lower())
        word_counts = Counter(doc_words)
        
        # Calculate similarity based on common words
        common_words = query_words.intersection(set(doc_words))
        if common_words:
            # Calculate score based on word frequency and commonality
            score = sum(word_counts[word] for word in common_words) / len(doc_words)
            
            # Create topic text from most frequent words
            top_words = [word for word, _ in word_counts.most_common(5)]
            topic_text = " ".join(top_words)
            
            results.append({
                "rank": len(results) + 1,
                "doc_index": doc_idx,
                "score": float(score),
                "text": topic_text
            })
    
    # Sort by score and apply top_k
    results.sort(key=lambda x: x["score"], reverse=True)
    if top_k is not None:
        results = results[:top_k]
        
    return results


if __name__ == "__main__":
    print("Using sample documents...")
    # docs: List[str] = [
    #     "The stock market crashed today as tech stocks took a hit.",
    #     "A new study shows the health benefits of a Mediterranean diet.",
    #     "NASA plans to launch a new satellite to monitor climate change.",
    #     "Python is a popular programming language for data science.",
    #     "The local team won the championship after a thrilling final."
    # ]
    chunks = load_file("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/search/llama_cpp/generated/run_vector_search/chunked_128_32/chunks.json")
    docs: List[str] = [chunk["content"] for chunk in chunks]

    model = "embeddinggemma"
    query = "How to change max depth?"
    top_k = 10
    
    print("Searching topics...")
    topic_results: List[Topic] = search_topics(
        query, docs, model, top_k
    )
    print(f"\nSearched Topics ({len(topic_results)})")
    save_file({
        "query": query,
        "count": len(topic_results),
        "topics": topic_results,
    }, f"{OUTPUT_DIR}/topic_results.json")