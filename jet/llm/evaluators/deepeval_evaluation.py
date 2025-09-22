import os
from typing import List

from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    GEval
)


def create_test_case(
    input_query: str,
    rag_chain,
    expected_output: str,
    k: int = 3
) -> LLMTestCase:
    """
    Creates a DeepEval test case for RAG evaluation.
    
    Args:
        input_query: User input query.
        rag_chain: RetrievalQA chain.
        expected_output: Mock expected output.
        k: Number of retrieved contexts.
    
    Returns:
        LLMTestCase instance.
    """
    # RAG output
    actual_output = rag_chain.invoke(input_query)['result']

    # contexts used from the retriever
    retrieved_contexts: List[str] = []
    for el in range(0, k):
        retrieved_contexts.append(rag_chain.invoke(input_query)['source_documents'][el].page_content)

    # create test case
    test_case = LLMTestCase(
        input=input_query,
        actual_output=actual_output,
        retrieval_context=retrieved_contexts,
        expected_output=expected_output
    )

    return test_case


def evaluate_retriever(test_case: LLMTestCase) -> None:
    """
    Evaluates the retriever component using Contextual Precision, Recall, and Relevancy metrics.
    
    Args:
        test_case: LLMTestCase instance.
    """
    # Initialize metrics
    contextual_precision = ContextualPrecisionMetric()
    contextual_recall = ContextualRecallMetric()
    contextual_relevancy = ContextualRelevancyMetric()

    # compute contextual precision and print results
    contextual_precision.measure(test_case)
    print("Score: ", contextual_precision.score)
    print("Reason: ", contextual_precision.reason)

    # compute contextual recall and print results
    contextual_recall.measure(test_case)
    print("Score: ", contextual_recall.score)
    print("Reason: ", contextual_recall.reason)

    # compute relevancy and print results
    contextual_relevancy.measure(test_case)
    print("Score: ", contextual_relevancy.score)
    print("Reason: ", contextual_relevancy.reason)

    # run all metrics with 'evaluate' function
    evaluate(
        test_cases=[test_case],
        metrics=[contextual_precision, contextual_recall, contextual_relevancy]
    )


def evaluate_generator(test_case: LLMTestCase) -> None:
    """
    Evaluates the generator component using Answer Relevancy and Faithfulness metrics.
    
    Args:
        test_case: LLMTestCase instance.
    """
    answer_relevancy = AnswerRelevancyMetric()
    faithfulness = FaithfulnessMetric()

    # compute answer relevancy and print results
    answer_relevancy.measure(test_case)
    print("Score: ", answer_relevancy.score)
    print("Reason: ", answer_relevancy.reason)

    # compute faithfulness and print results
    faithfulness.measure(test_case)
    print("Score: ", faithfulness.score)
    print("Reason: ", faithfulness.reason)

    # run all metrics with 'evaluate' function
    evaluate(
        test_cases=[test_case],
        metrics=[answer_relevancy, faithfulness]
    )


def evaluate_technical_language(test_case: LLMTestCase) -> None:
    """
    Custom GEval evaluation for technical language in the output.
    
    Args:
        test_case: LLMTestCase instance.
    """
    # create evaluation for technical language
    tech_eval = GEval(
        name="Technical Language",
        criteria="Determine how technically written the actual output is",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT]
    )

    # run evaluation
    tech_eval.measure(test_case)
    print("Score: ", tech_eval.score)
    print("Reason: ", tech_eval.reason)


if __name__ == "__main__":
    # Set your OpenAI API key
    os.environ["OPENAI_API_KEY"] = "your-api-key-here"  # Replace with actual key

    # define user query
    input_query = 'What are the most recent advancements in computer vision?'

    # expected output (example)
    expected_output = (
        'Recent advancements in computer vision include Vision-Language Models (VLMs) that merge vision and language, '
        'Neural Radiance Fields (NeRFs) for 3D scene generation, and powerful Diffusion Models and Generative AI for '
        'creating realistic visuals. Other key areas are Edge AI for real-time processing, enhanced 3D vision techniques '
        'like NeRFs and Visual SLAM, advanced self-supervised learning methods, deepfake detection systems, and increased '
        'focus on Ethical AI and Explainable AI (XAI) to ensure fairness and transparency.'
    )

    # Assume rag is loaded from rag_solution.py (e.g., from rag_solution import setup_rag_pipeline, create_vector_store; etc.)
    # For full run: load ai_search, vector_store, rag as in rag_solution.py
    # test_case = create_test_case(input_query, rag, expected_output)

    # Example (uncomment after loading rag):
    # test_case = create_test_case(input_query, rag, expected_output)
    # evaluate_retriever(test_case)
    # evaluate_generator(test_case)
    # evaluate_technical_language(test_case)