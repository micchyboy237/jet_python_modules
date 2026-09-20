"""Custom DeepEval LLM adapter for llama.cpp via jet infrastructure.
Summary:
    This module provides a DeepEval-compatible LLM wrapper that delegates
    inference to jet.adapters.llama_cpp.llm_utils_observed. It reuses existing
    configuration and ensures all evaluation calls are fully observed/traced.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from typing import Any, Dict, Optional, Union

from deepeval.models import DeepEvalBaseLLM
from jet.adapters.llama_cpp.config import LLM_BASE_URL, LLM_MODEL
from jet.adapters.llama_cpp.llm_utils_observed import achat, chat
from jet.observability import llm_span, redact
from openinference.semconv.trace import SpanAttributes
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class LlamacppModel(DeepEvalBaseLLM):
    """DeepEval LLM adapter backed by jet's observed llama.cpp infrastructure."""

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 16384,
        top_p: float = 0.8,
        top_k: int = 20,
        repeat_penalty: float = 1.1,
        presence_penalty: float = 1.5,
        frequency_penalty: float = 0.0,
        seed: Optional[int] = None,
        enable_thinking: bool = False,
        project_name: str = "deepeval-judge",
        generation_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        self.model_name = model or LLM_MODEL
        self.base_url = base_url or LLM_BASE_URL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.repeat_penalty = repeat_penalty
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.seed = seed
        self.enable_thinking = enable_thinking
        self.project_name = project_name
        self.generation_kwargs = generation_kwargs or {}
        self.generation_kwargs.update(kwargs)
        super().__init__(model=self.model_name)

    def load_model(self, *args: Any, **kwargs: Any) -> "LlamacppModel":
        return self

    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Union[str, BaseModel]:
        """Synchronous generation with detailed observability.

        NOTE: DeepEval expects this to return ONLY the string/model, not a tuple.
        """
        messages = [{"role": "user", "content": prompt}]
        invocation_params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }

        with llm_span(
            name="deepeval.judge.generate",
            model_name=self.model_name,
            messages=messages,
            invocation_params=invocation_params,
            provider="llama_cpp",
        ) as span:
            start_time = time.perf_counter()
            result = chat(
                prompt_or_messages=prompt,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                top_k=self.top_k,
                repeat_penalty=self.repeat_penalty,
                presence_penalty=self.presence_penalty,
                frequency_penalty=self.frequency_penalty,
                seed=self.seed,
                enable_thinking=self.enable_thinking,
                response_format=schema,
                project_name=self.project_name,
                capture_content=True,
                **self.generation_kwargs,
            )
            elapsed = time.perf_counter() - start_time

            # Capture structured output validation status
            if schema and result.structured:
                span.set_attribute(
                    "llm.structured_output.success", result.structured.success
                )
                span.set_attribute(
                    "llm.structured_output.format", result.structured.format_used.value
                )
                if result.structured.error:
                    span.set_attribute(
                        "llm.structured_output.error", redact(result.structured.error)
                    )

            # Capture token usage and latency
            if result.usage:
                span.set_attribute(
                    SpanAttributes.LLM_TOKEN_COUNT_PROMPT,
                    result.usage.get("prompt_tokens", 0),
                )
                span.set_attribute(
                    SpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
                    result.usage.get("completion_tokens", 0),
                )
                span.set_attribute(
                    SpanAttributes.LLM_TOKEN_COUNT_TOTAL,
                    result.usage.get("total_tokens", 0),
                )

            span.set_attribute("llm.latency.total_s", round(elapsed, 4))

            # Return parsed object if schema is present, otherwise raw content
            # IMPORTANT: Do NOT return a tuple here for DeepEval compatibility
            if schema and result.structured and result.structured.success:
                return result.structured.parsed
            return result.content

    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Union[str, BaseModel]:
        """Asynchronous generation with detailed observability.

        NOTE: DeepEval expects this to return ONLY the string/model, not a tuple.
        """
        messages = [{"role": "user", "content": prompt}]
        invocation_params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }

        with llm_span(
            name="deepeval.judge.a_generate",
            model_name=self.model_name,
            messages=messages,
            invocation_params=invocation_params,
            provider="llama_cpp",
        ) as span:
            start_time = time.perf_counter()
            result = await achat(
                prompt_or_messages=prompt,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                top_k=self.top_k,
                repeat_penalty=self.repeat_penalty,
                presence_penalty=self.presence_penalty,
                frequency_penalty=self.frequency_penalty,
                seed=self.seed,
                enable_thinking=self.enable_thinking,
                response_format=schema,
                project_name=self.project_name,
                capture_content=True,
                **self.generation_kwargs,
            )
            elapsed = time.perf_counter() - start_time

            if schema and result.structured:
                span.set_attribute(
                    "llm.structured_output.success", result.structured.success
                )
                span.set_attribute(
                    "llm.structured_output.format", result.structured.format_used.value
                )
                if result.structured.error:
                    span.set_attribute(
                        "llm.structured_output.error", redact(result.structured.error)
                    )

            if result.usage:
                span.set_attribute(
                    SpanAttributes.LLM_TOKEN_COUNT_PROMPT,
                    result.usage.get("prompt_tokens", 0),
                )
                span.set_attribute(
                    SpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
                    result.usage.get("completion_tokens", 0),
                )
                span.set_attribute(
                    SpanAttributes.LLM_TOKEN_COUNT_TOTAL,
                    result.usage.get("total_tokens", 0),
                )

            span.set_attribute("llm.latency.total_s", round(elapsed, 4))

            if schema and result.structured and result.structured.success:
                return result.structured.parsed
            return result.content

    def get_model_name(self, *args: Any, **kwargs: Any) -> str:
        return f"{self.model_name} (llama.cpp)"

    def supports_temperature(self) -> bool:
        return True

    def supports_structured_outputs(self) -> bool:
        return True

    def supports_json_mode(self) -> bool:
        return True

    def supports_multimodal(self) -> bool:
        return False


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test LlamacppModel DeepEval adapter.")
    parser.add_argument("query", type=str, help="Prompt/query to send to the model.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Model ID override (default: {LLM_MODEL}).",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature."
    )
    parser.add_argument(
        "--async-mode", action="store_true", default=False, help="Use async a_generate."
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    args = _build_parser().parse_args()
    model = LlamacppModel(model=args.model, temperature=args.temperature)
    logger.info("🤖 Model: %s", model.get_model_name())
    if args.async_mode:
        response = asyncio.run(model.a_generate(args.query))
    else:
        response = model.generate(args.query)
    print(f"\n📋 Response ({len(str(response))} chars):\n{response}")
