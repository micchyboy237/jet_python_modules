"""
Thin client for a local llama.cpp server (llama-server).

Assumes the server is started with its OpenAI-compatible API, e.g.:

    llama-server -m your-model.gguf -c 8192 --port 8080

Two endpoints are used:
  - POST /v1/chat/completions  -> chat completion (OpenAI-compatible)
  - POST /tokenize             -> exact token count for a string (llama.cpp-native)

Using the server's own /tokenize endpoint (rather than a guess like
len(text)//4) matters here: token budgets are only meaningful if they're
measured with the same tokenizer the model actually uses.
"""

from __future__ import annotations

import requests
from dataclasses import dataclass
from typing import Optional


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LlamaCppClient:
    def __init__(self, base_url: str = "http://localhost:8080", timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def count_tokens(self, text: str) -> int:
        """Exact token count via the server's tokenizer."""
        if not text:
            return 0
        resp = requests.post(
            f"{self.base_url}/tokenize",
            json={"content": text},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return len(resp.json()["tokens"])

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.3,
        stop: Optional[list[str]] = None,
    ) -> ChatResult:
        """
        Single-turn (or multi-turn) chat completion.
        `messages` is a plain OpenAI-style list: [{"role": "...", "content": "..."}]
        """
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop:
            payload["stop"] = stop

        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        # llama.cpp usually reports usage; fall back to local tokenizer if absent.
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if prompt_tokens is None:
            prompt_tokens = sum(self.count_tokens(m["content"]) for m in messages)
        if completion_tokens is None:
            completion_tokens = self.count_tokens(text)

        return ChatResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
