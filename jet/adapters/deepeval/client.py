import asyncio

from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from openai import AsyncOpenAI, OpenAI


class CustomOpenAIClient(DeepEvalBaseLLM):
    """
    A fully custom LLM client for DeepEval that wraps any
    OpenAI-compatible API (e.g., vLLM, LM Studio, Together AI).
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize both sync and async clients
        self._sync_client = OpenAI(base_url=base_url, api_key=api_key)
        self._async_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    def load_model(self):
        """Required by DeepEvalBaseLLM. Returns the underlying client."""
        return self._sync_client

    def generate(self, prompt: str) -> str:
        """Synchronous generation - used when running tests sequentially."""
        try:
            response = self._sync_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"CustomOpenAIClient sync generation failed: {e}")

    async def a_generate(self, prompt: str) -> str:
        """Async generation - REQUIRED for parallel metric evaluation."""
        try:
            response = await self._async_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"CustomOpenAIClient async generation failed: {e}")

    def get_model_name(self) -> str:
        return self.model_name


# ─── USAGE IN EVALUATION ───────────────────────────────────────────────


def test_rag_with_custom_judge():
    # 1. Instantiate your custom judge model
    judge_model = CustomOpenAIClient(
        model_name="meta-llama/Llama-3-8b-instruct",
        base_url="http://localhost:8000/v1",  # e.g., vLLM server
        api_key="EMPTY",  # not needed for local
        temperature=0.0,  # deterministic judging
    )

    # 2. Define test case from your RAG pipeline
    test_case = LLMTestCase(
        input="What are the side effects of metformin?",
        actual_output="Common side effects include nausea, diarrhea, and stomach upset.",
        retrieval_context=[
            "Metformin commonly causes gastrointestinal issues such as nausea and diarrhea.",
            "Lactic acidosis is a rare but serious side effect of metformin.",
            "Metformin is a first-line treatment for type 2 diabetes.",
        ],
    )

    # 3. Pass custom model to ALL metrics
    metrics = [
        FaithfulnessMetric(threshold=0.7, model=judge_model),
        AnswerRelevancyMetric(threshold=0.7, model=judge_model),
    ]

    # 4. Run assertion (raises AssertionError if any metric fails)
    assert_test(test_case, metrics)


# ─── BATCH EVALUATION WITH ASYNC PARALLELISM ──────────────────────────


async def batch_evaluate():
    """Demonstrates async parallel evaluation across multiple test cases."""
    judge_model = CustomOpenAIClient(
        model_name="meta-llama/Llama-3-8b-instruct",
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
    )

    test_cases = [
        LLMTestCase(
            input="What is photosynthesis?",
            actual_output="Photosynthesis converts sunlight into chemical energy in plants.",
            retrieval_context=[
                "Plants use chlorophyll to absorb light during photosynthesis."
            ],
        ),
        LLMTestCase(
            input="Who wrote Hamlet?",
            actual_output="Hamlet was written by William Shakespeare around 1600.",
            retrieval_context=[
                "William Shakespeare authored Hamlet, likely between 1599 and 1601."
            ],
        ),
    ]

    metric = FaithfulnessMetric(threshold=0.7, model=judge_model)

    # a_measure runs all evaluations concurrently using a_generate()
    results = await asyncio.gather(*[metric.a_measure(tc) for tc in test_cases])

    for i, score in enumerate(results):
        print(
            f"Test {i + 1}: Faithfulness = {score:.2f} | {'PASS' if score >= 0.7 else 'FAIL'}"
        )


if __name__ == "__main__":
    # Single test
    test_rag_with_custom_judge()

    # Batch async evaluation
    asyncio.run(batch_evaluate())
