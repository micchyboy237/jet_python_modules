import os
from typing import Optional
from jet.llm.models import OLLAMA_EMBED_MODELS, OLLAMA_MODEL_NAMES
import numpy as np
import nltk
from jet.features.eval_search_and_chat import evaluate_llm_response, save_output
from jet.features.queue_config import eval_queue
from nltk.sentiment.vader import SentimentIntensityAnalyzer

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    nltk.download('vader_lexicon')


def evaluate_response_tone(query: str, response: str, output_dir: str):
    """Evaluates the sentiment/tone of the LLM response."""
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


def enqueue_evaluation_task(query: str, response: str, context: str, embed_model: OLLAMA_EMBED_MODELS = "mxbai-embed-large", llm_model: OLLAMA_MODEL_NAMES = "gemma3:4b", output_dir: Optional[str] = None):
    job = eval_queue.enqueue(
        evaluate_llm_response,
        query=query,
        response=response,
        context=context,
        output_dir=output_dir,
        embed_model=embed_model,
        llm_model=llm_model
    )
    print(f"Enqueued LLM response evaluation job with ID: {job.id}")
    return job


def enqueue_tone_evaluation_task(query: str, response: str, output_dir: str):
    """Enqueues a job to evaluate the tone of the LLM response."""
    job = eval_queue.enqueue(
        'jet.features.eval_tasks.evaluate_response_tone',  # Enqueue by module path
        query=query,
        response=response,
        output_dir=output_dir
    )
    print(f"Enqueued tone evaluation job with ID: {job.id}")
    return job


if __name__ == "__main__":
    import os
    query = "What is the capital of France?"
    response = "The capital of France is Paris."
    context = "France is a country in Western Europe, and its capital city is Paris."
    embed_model = "mxbai-embed-large"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    llm_model = "llama3.1"
    os.makedirs(output_dir, exist_ok=True)
    enqueue_evaluation_task(query, response, context,
                            embed_model, llm_model, output_dir)
    enqueue_tone_evaluation_task(query, response, output_dir)
