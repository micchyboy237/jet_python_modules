from jet.logger import logger
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
import time
from jet.llm.models import OLLAMA_EMBED_MODELS, OLLAMA_MODEL_NAMES
from jet.llm.utils.embeddings import get_embedding_function
import numpy as np
from scipy.stats import entropy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from jet.file.utils import save_file
from llama_index.core.schema import Document, NodeWithScore
from collections import Counter
from urllib.parse import urlparse
import nltk
import math
import psutil
from typing import Any, List, Dict, Optional
from jet.llm.evaluators.context_relevancy_evaluator import evaluate_context_relevancy
from jet.llm.evaluators.answer_relevancy_evaluator import evaluate_answer_relevancy

# Download NLTK data for tokenization
nltk.download('punkt', quiet=True)

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except:
    nltk.download('vader_lexicon')


def save_output(data: Any, filepath: str) -> None:
    """Helper to save data to file with error handling."""
    try:
        save_file(data, filepath)
    except Exception as e:
        print(f"Error saving {filepath}: {str(e)}")


def reconstruct_nodes(nodes: List[Dict]) -> List[NodeWithScore]:
    """Helper to reconstruct NodeWithScore objects from serialized data."""
    return [
        NodeWithScore(
            node=Document(text=node["text"], metadata=node["metadata"]),
            score=node["score"]
        ) for node in nodes
    ]


def evaluate_search_results(search_results: List[Dict[str, str]], query: str, output_dir: str) -> dict:
    """Evaluate the quality of search results."""
    evaluation = {
        "total_results": len(search_results),
        "unique_domains": 0,
        "url_diversity_score": 0.0,
        "keyword_overlap_score": 0.0,
    }

    # Extract domains from URLs
    domains = [urlparse(result.get("url", "")
                        ).netloc for result in search_results]
    domain_counts = Counter(domains)

    # Calculate unique domains and diversity
    evaluation["unique_domains"] = len(domain_counts)
    total_urls = len(domains)
    if total_urls > 0:
        diversity_score = -sum(
            (count / total_urls) * math.log2(count / total_urls)
            for count in domain_counts.values()
            if count > 0
        ) / (total_urls or 1)
        evaluation["url_diversity_score"] = diversity_score

    # Keyword overlap score
    query_tokens = set(nltk.word_tokenize(query.lower()))
    if query_tokens:
        overlap_scores = [
            len(query_tokens.intersection(set(nltk.word_tokenize((result.get(
                "title", "") + " " + result.get("snippet", "")).lower())))) / len(query_tokens)
            for result in search_results
        ]
        evaluation["keyword_overlap_score"] = np.mean(
            overlap_scores) if overlap_scores else 0.0

    # Save evaluation
    save_output(evaluation, os.path.join(
        output_dir, "search_results_evaluation.json"))
    return evaluation


def evaluate_html_processing(query_scores: list, reranked_nodes: list, grouped_nodes: list, output_dir: str) -> dict:
    """Evaluate HTML processing and reranking."""
    evaluation = {
        "score_distribution_entropy": 0.0,
        "node_relevance_score": 0.0,
        "grouped_nodes_coherence": 0.0,
        "ndcg_at_5": 0.0,
    }

    # Reconstruct NodeWithScore objects
    reranked_nodes = reconstruct_nodes(reranked_nodes)
    grouped_nodes = reconstruct_nodes(grouped_nodes)

    # Score distribution entropy
    if query_scores:
        normalized_scores = np.array(
            query_scores) / (np.sum(query_scores) + 1e-10)
        evaluation["score_distribution_entropy"] = entropy(normalized_scores)

    # Node relevance (average score of top reranked nodes)
    if reranked_nodes:
        top_scores = [node.score for node in reranked_nodes[:5]]
        evaluation["node_relevance_score"] = np.mean(
            top_scores) if top_scores else 0.0

    # Grouped nodes coherence (cosine similarity between node texts)
    if grouped_nodes and len(grouped_nodes) > 1:
        vectorizer = TfidfVectorizer()
        node_texts = [node.text for node in grouped_nodes]
        tfidf_matrix = vectorizer.fit_transform(node_texts)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        similarities = similarity_matrix[np.triu_indices(len(node_texts), k=1)]
        evaluation["grouped_nodes_coherence"] = np.mean(
            similarities) if similarities.size > 0 else 0.0

    # NDCG@5
    if reranked_nodes:
        scores = [node.score for node in reranked_nodes[:5]]
        ideal_scores = sorted(scores, reverse=True)
        dcg = sum(score / math.log2(i + 2) for i, score in enumerate(scores))
        idcg = sum(score / math.log2(i + 2)
                   for i, score in enumerate(ideal_scores))
        evaluation["ndcg_at_5"] = dcg / idcg if idcg > 0 else 0.0

    # Save evaluation
    save_output(evaluation, os.path.join(
        output_dir, "html_processing_evaluation.json"))
    return evaluation


def evaluate_llm_response(query: str, context: str, response: str, embed_model: OLLAMA_EMBED_MODELS = "mxbai-embed-large", llm_model: OLLAMA_MODEL_NAMES = "gemma3:4b", output_dir: Optional[str] = None) -> dict:
    """Evaluate LLM response for relevance and coherence."""
    evaluation = {
        "query_response_similarity": 0.0,
        "context_response_similarity": 0.0,
        "response_coherence_score": 0.0,
        "context_relevancy": {},
        "answer_relevancy": {}
    }

    # Early return for empty inputs
    if not response or not query or not context:
        save_output(evaluation, os.path.join(
            output_dir, "llm_response_evaluation.json"))
        return evaluation

    # Evaluate context relevancy
    context_result = evaluate_context_relevancy(
        model=llm_model,
        query=query,
        contexts=context
    )
    evaluation["context_relevancy"] = {
        "passing": context_result.passing,
        "score": context_result.score,
        "feedback": context_result.feedback,
        "details": context_result.details,
    }

    # Evaluate answer relevancy
    answer_result = evaluate_answer_relevancy(
        model=llm_model,
        query=query,
        response=response
    )
    evaluation["answer_relevancy"] = {
        "passing": answer_result.passing,
        "score": answer_result.score,
        "feedback": answer_result.feedback,
        "details": answer_result.details,
    }

    embed_func = get_embedding_function(embed_model)

    # Get embeddings
    query_embedding = np.array(embed_func(query))
    response_embedding = np.array(embed_func(response))
    context_embedding = np.array(embed_func(context))

    # Compute cosine similarities
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

    evaluation["query_response_similarity"] = cosine_similarity(
        query_embedding, response_embedding)
    evaluation["context_response_similarity"] = cosine_similarity(
        context_embedding, response_embedding)

    # Response coherence (pairwise similarity of sentences)
    sentences = nltk.sent_tokenize(response)
    print(f"Number of sentences: {len(sentences)}")
    if len(sentences) > 1:
        sentence_embeddings = [np.array(embed_func(s)) for s in sentences]
        similarities = [
            cosine_similarity(sentence_embeddings[i], sentence_embeddings[j])
            for i in range(len(sentences))
            for j in range(i + 1, len(sentences))
        ]
        evaluation["response_coherence_score"] = np.mean(
            similarities) if similarities else 0.0
    else:
        evaluation["response_coherence_score"] = 1.0  # Single sentence case

    evaluation = {
        "llm_model": llm_model,
        "embed_model": embed_model,
        **evaluation,
    }

    # Save evaluation
    if output_dir:
        save_output(evaluation, os.path.join(
            output_dir, "evaluation", "eval_llm_context_and_response.json"))

    return evaluation


def evaluate_pipeline(start_time: float, output_dir: str, error_count: int) -> dict:
    """Evaluate overall pipeline performance."""
    evaluation = {
        "latency_seconds": time.time() - start_time,
        "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
        "cpu_usage_percent": psutil.cpu_percent(interval=None),
        "error_count": error_count,
        "error_rate": error_count / (error_count + 1)
    }

    # Save evaluation
    save_output(evaluation, os.path.join(
        output_dir, "pipeline_evaluation.json"))
    return evaluation


def evaluate_response_tone(query: str, response: str, output_dir: str):
    """Evaluates the sentiment/tone of the LLM response."""
    try:
        analyzer = SentimentIntensityAnalyzer()
        vs = analyzer.polarity_scores(response)
        tone = "Neutral"
        if vs['compound'] >= 0.05:
            tone = "Positive"
        elif vs['compound'] <= -0.05:
            tone = "Negative"
        output_path = os.path.join(output_dir, f"tone_evaluation.json")
        save_output({"query": query, "response": response,
                     "tone": tone, "sentiment_scores": vs}, output_path)
        print(f"Tone evaluation results saved to: {output_path}")
    except Exception as e:
        logger.error(f"Error in evaluate_response_tone: {e}", exc_info=True)
        raise
