"""LlamaCppLLM: generic wrapper for llama.cpp LLM server (OpenAI chat compatible)."""
import os
from typing import List, Dict, Optional
from openai import OpenAI

class LlamaCppLLM:
    """Reusable LLM generator - flexible messages, temperature etc. No business prompts here."""

    def __init__(self, llm_url: Optional[str] = None):
        url = llm_url or os.getenv("LLAMA_CPP_LLM_URL")
        if not url:
            raise ValueError("LLAMA_CPP_LLM_URL env var required (or pass llm_url)")
        self.client = OpenAI(
            base_url=url.rstrip("/") + "/v1" if not url.endswith("/v1") else url,
            api_key="sk-no-key-required",
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
