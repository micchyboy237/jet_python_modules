"""02_batch_evaluation.py - Evaluating multiple responses in a loop."""

from jet.adapters.llama_cpp.tasks.rag.eval.faithfulness_relevancy_evaluator import (
    evaluate_faithfulness_llamacpp,
)
from jet.logger import logger


def main():
    logger.info("Starting Batch Faithfulness Evaluation Demo")

    test_cases = [
        {
            "query": "Who wrote Hamlet?",
            "response": "William Shakespeare wrote Hamlet.",
            "contexts": ["Hamlet is a tragedy written by William Shakespeare."],
            "expected": True,
        },
        {
            "query": "What is the speed of light?",
            "response": "The speed of light is approximately 300,000 km/s.",
            "contexts": [
                "Light travels at about 186,000 miles per second in a vacuum."
            ],
            "expected": True,  # 186k miles is approx 300k km
        },
        {
            "query": "What is the capital of Japan?",
            "response": "Tokyo is the capital of Japan.",
            "contexts": [
                "Kyoto was the imperial capital of Japan for over a thousand years."
            ],
            "expected": False,  # Context doesn't support current capital
        },
    ]

    for i, case in enumerate(test_cases):
        print(f"\n--- Test Case {i + 1} ---")
        print(f"Query: {case['query']}")
        print(f"Response: {case['response']}")

        result = evaluate_faithfulness_llamacpp(
            query=case["query"],
            response=case["response"],
            contexts=case["contexts"],
        )

        status = "✅ PASS" if result.passing == case["expected"] else "❌ FAIL"
        print(
            f"Result: {status} (Passing: {result.passing}, Expected: {case['expected']})"
        )
        print(f"Feedback: {result.feedback}")


if __name__ == "__main__":
    main()
