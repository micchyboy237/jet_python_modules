# jet/features/eval_tasks.py

import os
import numpy as np
import nltk
from jet.llm.utils.embeddings import get_embedding_function
from jet.features.eval_search_and_chat import evaluate_context_relevancy, evaluate_answer_relevancy, evaluate_llm_response, save_output
from jet.features.eval_worker import eval_queue


def enqueue_evaluation_task(query: str, response: str, context: str, embed_model: str, output_dir: str, llm_model: str = "llama3.1"):
    job = eval_queue.enqueue(
        evaluate_llm_response,
        query=query,
        response=response,
        context=context,
        output_dir=output_dir,
        embed_model=embed_model,
        llm_model=llm_model
    )
    print(f"Enqueued job with ID: {job.id}")
    return job


if __name__ == "__main__":
    import os

    query = "What is the capital of France?"
    response = "The capital of France is Paris."
    context = "France is a country in Western Europe, and its capital city is Paris."
    embed_model = "mxbai-embed-large"
    output_dir = "./evaluation_results"
    llm_model = "llama3.1"

    os.makedirs(output_dir, exist_ok=True)
    enqueue_evaluation_task(query, response, context,
                            embed_model, output_dir, llm_model)
