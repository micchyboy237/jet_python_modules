import asyncio

from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from jet.adapters.llama_cpp.config import LLM_BASE_URL, LLM_MODEL
from openai import AsyncOpenAI, OpenAI
from jet.adapters.llama_cpp.llm_utils import chat, achat


class CustomOpenAIClient(DeepEvalBaseLLM):
    """
    A fully custom LLM client for DeepEval that wraps any
    OpenAI-compatible API (e.g., vLLM, LM Studio, Together AI).
    Now supports configurable OpenAI API args.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        top_p: float = 0.95,
        presence_penalty: float = 1.5,
        frequency_penalty: float = 0.0,
        stop: list[str] | None = None,
        seed: int | None = None,
        logit_bias: dict[str, int] | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        response_format: dict | None = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.stop = stop
        self.seed = seed
        self.logit_bias = logit_bias
        self.tools = tools
        self.tool_choice = tool_choice
        self.response_format = response_format

        # Initialize both sync and async clients
        self._sync_client = OpenAI(
            base_url=base_url,
            api_key="not-needed",
            timeout=120.0,
            max_retries=3,
        )
        self._async_client = AsyncOpenAI(
            base_url=base_url,
            api_key="not-needed",
            timeout=120.0,
            max_retries=3,
        )

    def load_model(self):
        """Required by DeepEvalBaseLLM. Returns the underlying client."""
        return self._sync_client

    def _build_chat_args(self, prompt: str) -> dict:
        args = dict(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
        )
        if self.stop is not None:
            args["stop"] = self.stop
        if self.seed is not None:
            args["seed"] = self.seed
        if self.logit_bias is not None:
            args["logit_bias"] = self.logit_bias
        if self.tools is not None:
            args["tools"] = self.tools
        if self.tool_choice is not None:
            args["tool_choice"] = self.tool_choice
        if self.response_format is not None:
            args["response_format"] = self.response_format
        return args

    def generate(self, prompt: str) -> str:
        """Synchronous generation - used when running tests sequentially."""
        try:
            chat_args = self._build_chat_args(prompt)
            response = self._sync_client.chat.completions.create(**chat_args)
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"CustomOpenAIClient sync generation failed: {e}")

    async def a_generate(self, prompt: str) -> str:
        """Async generation - REQUIRED for parallel metric evaluation."""
        try:
            chat_args = self._build_chat_args(prompt)
            response = await self._async_client.chat.completions.create(**chat_args)
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"CustomOpenAIClient async generation failed: {e}")

    def get_model_name(self) -> str:
        return self.model_name


# ─── USAGE IN EVALUATION ───────────────────────────────────────────────


def test_rag_with_custom_judge():
    # 1. Instantiate your custom judge model
    judge_model = CustomOpenAIClient(
        model_name=LLM_MODEL,
        base_url=LLM_BASE_URL,  # e.g., vLLM server
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
        model_name=LLM_MODEL,
        base_url=LLM_BASE_URL,
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
