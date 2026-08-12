import asyncio
import logging
from typing import Any, Callable

from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from jet.adapters.llama_cpp.config import LLM_BASE_URL, LLM_MODEL
from jet.adapters.llama_cpp.llm_utils import achat, chat
from openai import AsyncOpenAI, OpenAI

logger = logging.getLogger(__name__)

# Default Phoenix project for DeepEval judge traces
DEFAULT_PROJECT_NAME = "deepeval-judge"


class CustomOpenAIClient(DeepEvalBaseLLM):
    """
    DeepEval LLM client that reuses jet.adapters.llama_cpp.llm_utils.chat / achat.

    This gives DeepEval the same observability, streaming, and llama.cpp
    sampling controls as the rest of the stack, without duplicating client logic.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        *,
        project_name: str = DEFAULT_PROJECT_NAME,
        capture_content: bool = True,
        phoenix_url: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.0,
        repeat_penalty: float = 1.1,
        presence_penalty: float = 1.5,
        frequency_penalty: float = 0.0,
        stop: list[str] | None = None,
        seed: int | None = None,
        logit_bias: dict[str, int] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        tool_registry: dict[str, Callable[..., Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        enable_thinking: bool = False,
        max_tool_rounds: int = 10,
        extra_body_params: dict[str, Any] | None = None,
        session_id: str | None = None,
        client: OpenAI | None = None,
        async_client: AsyncOpenAI | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        self.model_name = model_name
        self.base_url = base_url or LLM_BASE_URL
        self.project_name = project_name
        self.capture_content = capture_content
        self.phoenix_url = phoenix_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.repeat_penalty = repeat_penalty
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.stop = stop
        self.seed = seed
        self.logit_bias = logit_bias
        self.tools = tools
        self.tool_choice = tool_choice
        self.tool_registry = tool_registry
        self.response_format = response_format
        self.enable_thinking = enable_thinking
        self.max_tool_rounds = max_tool_rounds
        self.extra_body_params = extra_body_params
        self.session_id = session_id

        # A: build once (or use injected clients)
        self._sync_client = client or OpenAI(
            base_url=self.base_url,
            api_key="not-needed",
            timeout=timeout,
            max_retries=max_retries,
        )
        self._async_client = async_client or AsyncOpenAI(
            base_url=self.base_url,
            api_key="not-needed",
            timeout=timeout,
            max_retries=max_retries,
        )

        logger.info(
            "CustomOpenAIClient init | model=%s project=%s base_url=%s "
            "temp=%.2f max_tokens=%d",
            self.model_name,
            self.project_name,
            self.base_url,
            self.temperature,
            self.max_tokens,
        )

    def load_model(self):
        """Required by DeepEvalBaseLLM. Returns the sync OpenAI client if present."""
        return self._sync_client

    def _common_kwargs(self, prompt: str) -> dict[str, Any]:
        """Shared kwargs for both chat and achat."""
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "model": self.model_name,
            "project_name": self.project_name,
            "capture_content": self.capture_content,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "repeat_penalty": self.repeat_penalty,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "stop": self.stop,
            "seed": self.seed,
            "logit_bias": self.logit_bias,
            "tools": self.tools,
            "tool_choice": self.tool_choice,
            "tool_registry": self.tool_registry,
            "response_format": self.response_format,
            "enable_thinking": self.enable_thinking,
            "max_tool_rounds": self.max_tool_rounds,
            "extra_body_params": self.extra_body_params,
            "session_id": self.session_id,
        }
        if self.phoenix_url is not None:
            kwargs["phoenix_url"] = self.phoenix_url
        return kwargs

    def generate(self, prompt: str) -> str:
        """Synchronous generation used by sequential DeepEval runs."""
        logger.debug(
            "generate | project=%s model=%s prompt_len=%d",
            self.project_name,
            self.model_name,
            len(prompt),
        )
        try:
            kwargs = self._common_kwargs(prompt)
            kwargs["client"] = self._sync_client  # always pass wired client
            result = chat(**kwargs)
            content = (result.content or "").strip()
            logger.info(
                "generate done | chars=%d finish_reason=%s has_tools=%s",
                len(content),
                result.finish_reason,
                result.has_tool_calls,
            )
            return content
        except Exception as e:
            logger.exception("CustomOpenAIClient.generate failed")
            raise RuntimeError(f"CustomOpenAIClient sync generation failed: {e}") from e

    async def a_generate(self, prompt: str) -> str:
        """Async generation required for parallel metric evaluation."""
        logger.debug(
            "a_generate | project=%s model=%s prompt_len=%d",
            self.project_name,
            self.model_name,
            len(prompt),
        )
        try:
            kwargs = self._common_kwargs(prompt)
            kwargs["client"] = self._async_client  # always pass wired client
            result = await achat(**kwargs)
            content = (result.content or "").strip()
            logger.info(
                "a_generate done | chars=%d finish_reason=%s has_tools=%s",
                len(content),
                result.finish_reason,
                result.has_tool_calls,
            )
            return content
        except Exception as e:
            logger.exception("CustomOpenAIClient.a_generate failed")
            raise RuntimeError(
                f"CustomOpenAIClient async generation failed: {e}"
            ) from e

    def get_model_name(self) -> str:
        return self.model_name


def test_rag_with_custom_judge():
    judge_model = CustomOpenAIClient(
        model_name=LLM_MODEL,
        base_url=LLM_BASE_URL,
        temperature=0.0,
        project_name="deepeval-rag-judge",
    )
    test_case = LLMTestCase(
        input="What are the side effects of metformin?",
        actual_output="Common side effects include nausea, diarrhea, and stomach upset.",
        retrieval_context=[
            "Metformin commonly causes gastrointestinal issues such as nausea and diarrhea.",
            "Lactic acidosis is a rare but serious side effect of metformin.",
            "Metformin is a first-line treatment for type 2 diabetes.",
        ],
    )
    metrics = [
        FaithfulnessMetric(threshold=0.7, model=judge_model),
        AnswerRelevancyMetric(threshold=0.7, model=judge_model),
    ]
    assert_test(test_case, metrics)


async def batch_evaluate():
    """Demonstrates async parallel evaluation across multiple test cases."""
    judge_model = CustomOpenAIClient(
        model_name=LLM_MODEL,
        base_url=LLM_BASE_URL,
        project_name="deepeval-batch-judge",
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
    results = await asyncio.gather(*[metric.a_measure(tc) for tc in test_cases])
    for i, score in enumerate(results):
        print(
            f"Test {i + 1}: Faithfulness = {score:.2f} | "
            f"{'PASS' if score >= 0.7 else 'FAIL'}"
        )


if __name__ == "__main__":
    test_rag_with_custom_judge()
    asyncio.run(batch_evaluate())
