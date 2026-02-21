"""LlamaCppLLM: generic wrapper for llama.cpp LLM server (OpenAI chat compatible)."""

import os
from typing import Dict, List

from jet.adapters.llama_cpp.types import LLAMACPP_LLM_KEYS
from openai import OpenAI

DEFAULT_LLM_URL = os.getenv("LLAMA_CPP_LLM_URL")
DEFAULT_MODEL_NAME: LLAMACPP_LLM_KEYS = os.getenv("LLAMA_CPP_LLM_MODEL")


class LlamaCppLLM:
    """Reusable LLM generator - flexible messages, temperature etc. No business prompts here."""

    def __init__(
        self, model: LLAMACPP_LLM_KEYS = DEFAULT_MODEL_NAME, url: str = DEFAULT_LLM_URL
    ):
        self.url = url
        self.model = model
        if not url:
            raise ValueError("LLAMA_CPP_LLM_URL env var required (or pass url)")
        self.client = OpenAI(
            base_url=url.rstrip("/") + "/v1" if not url.endswith("/v1") else url,
            api_key="sk-no-key-required",
            timeout=300.0,
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Generic chat completion - returns clean string."""
        response = self.client.chat.completions.create(
            model="dummy",  # ignored by server
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()
