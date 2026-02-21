"""LlamaCppLLM: generic wrapper for llama.cpp LLM server (OpenAI chat compatible)."""

import os
from typing import Dict, List

from jet.adapters.llama_cpp.tokens import count_tokens
from jet.adapters.llama_cpp.types import LLAMACPP_LLM_KEYS
from jet.libs.openai_python.utils import save_request_llm_call, save_response_llm_call
from jet.logger import logger
from openai import OpenAI, Stream
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, Choice
from rich.console import Console

console = Console()

DEFAULT_LLM_URL = os.getenv("LLAMA_CPP_LLM_URL")
DEFAULT_MODEL_NAME: LLAMACPP_LLM_KEYS = os.getenv("LLAMA_CPP_LLM_MODEL")


class LlamaCppLLM:
    """Reusable LLM generator - flexible messages, temperature etc. No business prompts here."""

    def __init__(
        self,
        model: LLAMACPP_LLM_KEYS = DEFAULT_MODEL_NAME,
        url: str = DEFAULT_LLM_URL,
        agent_name: str | None = None,
    ):
        self.url = url
        self.model = model
        self.agent_name = agent_name
        if not url:
            raise ValueError("LLAMA_CPP_LLM_URL env var required (or pass url)")
        self.client = OpenAI(
            base_url=url,
            api_key="sk-no-key-required",
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Generic chat completion - returns clean string."""
        input_tokens = count_tokens(messages, model=self.model)
        logger.info(f"Messages: {len(messages)}")
        logger.info(f"Input tokens: {input_tokens}")

        metadata = {
            "model": self.model,
            "token_counts": {"input_tokens": input_tokens},
            "generation_params": {
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        }
        save_request_llm_call(messages, metadata, agent_name=self.agent_name)

        stream: Stream[ChatCompletionChunk] = self.client.chat.completions.create(
            model=self.model,  # ignored by server
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )

        choices: list[Choice] = []
        response = ""
        usage = None
        for chunk in stream:
            if chunk.choices:
                choice = chunk.choices[0]
                delta = choice.delta
                content = delta.content or ""
                response += content

                choices.append(choice)
                logger.teal(content, flush=True, end="")

            # Optional: show final usage stats (tokens/s)
            if hasattr(chunk, "usage") and chunk.usage:
                usage = chunk.usage
                logger.info(f"\nUsage → {chunk.usage}")

        if usage:
            logger.success("--- Response Usage ---")
            logger.success(f"Input tokens: {usage.prompt_tokens}")
            logger.success(f"Output tokens: {usage.completion_tokens}")
            logger.success(f"Total tokens: {usage.total_tokens}")

        save_response_llm_call(
            response,
            usage,
            choices=choices,
            agent_name=self.agent_name,
        )

        return response.strip()
