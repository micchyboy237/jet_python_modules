"""Custom DeepEval LLM adapter for llama.cpp via jet infrastructure.

Summary:
    This module provides a DeepEval-compatible LLM wrapper that delegates
    inference to the jet.adapters.llama_cpp backend. It reuses existing
    configuration, client factories, and streaming utilities to ensure
    consistent behavior between evaluation metrics and production code.

Usage Examples:
    # Basic usage within a DeepEval metric
    from jet.adapters.deepeval.llamacpp_model import LlamacppModel
    model = LlamacppModel()
    response, cost = model.generate("What is the capital of France?")

    # With custom parameters
    model = LlamacppModel(
        model="qwen3.5-uncensored:2b",
        temperature=0.3,
        max_tokens=512
    )

    # CLI usage
    python -m jet.adapters.deepeval.llamacpp_model "Explain quantum entanglement" --temperature 0.7
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Any, Dict, Optional, Tuple, Union

from deepeval.models import DeepEvalBaseLLM
from jet.adapters.llama_cpp.config import LLM_BASE_URL, LLM_MODEL
from jet.adapters.llama_cpp.llm_utils import achat, chat
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class LlamacppModel(DeepEvalBaseLLM):
    """DeepEval LLM adapter backed by jet's llama.cpp infrastructure.

    Args:
        model: Model identifier. Defaults to LLAMA_CPP_LLM_MODEL env var.
        base_url: Server endpoint. Defaults to LLAMA_CPP_LLM_URL env var.
        temperature: Sampling temperature (0.0-2.0). Default 0.7.
        max_tokens: Maximum completion tokens. Default 16384.
        top_p: Nucleus sampling threshold. Default 0.8.
        top_k: Top-k sampling limit. Default 20.
        repeat_penalty: Repetition penalty. Default 1.1.
        presence_penalty: Presence penalty (-2.0 to 2.0). Default 1.5.
        frequency_penalty: Frequency penalty (-2.0 to 2.0). Default 0.0.
        seed: Random seed for reproducibility. None for random.
        enable_thinking: Enable reasoning/thinking tokens. Default False.
        generation_kwargs: Additional kwargs forwarded to jet.llm_utils.
    """

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
        self.generation_kwargs = generation_kwargs or {}

        # Merge any extra kwargs into generation_kwargs for flexibility
        self.generation_kwargs.update(kwargs)

        # Initialize parent with model name for tracing/naming
        super().__init__(model=self.model_name)

    def load_model(self, *args: Any, **kwargs: Any) -> "LlamacppModel":
        """No-op: jet manages client lifecycle internally."""
        return self

    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], float]:
        """Synchronous generation via jet.llm_utils.chat."""
        logger.debug(
            "LlamacppModel.generate called | model=%s | temp=%.2f | schema=%s",
            self.model_name,
            self.temperature,
            schema.__name__ if schema else None,
        )

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
            **self.generation_kwargs,
        )

        if schema and result.structured and result.structured.success:
            return result.structured.parsed, 0.0

        return result.content, 0.0

    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], float]:
        """Asynchronous generation via jet.llm_utils.achat."""
        logger.debug(
            "LlamacppModel.a_generate called | model=%s | temp=%.2f | schema=%s",
            self.model_name,
            self.temperature,
            schema.__name__ if schema else None,
        )

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
            **self.generation_kwargs,
        )

        if schema and result.structured and result.structured.success:
            return result.structured.parsed, 0.0

        return result.content, 0.0

    def get_model_name(self, *args: Any, **kwargs: Any) -> str:
        """Return display name for tracing and reporting."""
        return f"{self.model_name} (llama.cpp)"

    def supports_temperature(self) -> bool:
        return True

    def supports_structured_outputs(self) -> bool:
        return True

    def supports_json_mode(self) -> bool:
        return True

    def supports_multimodal(self) -> bool:
        # jet supports vision, but DeepEval's generate() signature is text-only.
        # Return False to prevent DeepEval from sending multimodal payloads
        # through the text-only generate path.
        return False


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test LlamacppModel DeepEval adapter against a local llama.cpp server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "query",
        type=str,
        help="Prompt/query to send to the model.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Model ID override (default: {LLM_MODEL}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Max completion tokens (default: 16384).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="Nucleus sampling top_p (default: 0.8).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=False,
        help="Enable reasoning/thinking tokens.",
    )
    parser.add_argument(
        "--async-mode",
        action="store_true",
        default=False,
        help="Use async a_generate instead of sync generate.",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = _build_parser().parse_args()

    model = LlamacppModel(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        seed=args.seed,
        enable_thinking=args.enable_thinking,
    )

    logger.info("🤖 Model: %s", model.get_model_name())
    logger.info("📝 Query: %s", args.query)

    if args.async_mode:
        response, cost = asyncio.run(model.a_generate(args.query))
    else:
        response, cost = model.generate(args.query)

    print("\n" + "=" * 60)
    print(f"📋 Response ({len(response)} chars):")
    print("=" * 60)
    print(response)
    print("=" * 60)
    logger.info(
        "💰 Cost: %.4f | Mode: %s", cost, "async" if args.async_mode else "sync"
    )
