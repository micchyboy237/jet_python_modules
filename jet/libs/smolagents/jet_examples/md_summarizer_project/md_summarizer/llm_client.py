"""LLM client abstraction for the summarization pipeline.

Two implementations:

- LlamaCppClient: talks to a real `llama-server` instance over its
  OpenAI-compatible HTTP API for completions, and its `/tokenize` endpoint for
  exact token counts (important -- word-count estimates drift from the real
  tokenizer, especially on markdown headers, code fences, and punctuation).

- MockLLMClient: a deterministic, no-network stand-in used by the demo so the
  pipeline logic (chunking, recursive reduce, tree walking) can be exercised
  and verified without a running model.
"""

import logging
import time
from abc import ABC, abstractmethod

import requests

logger = logging.getLogger("md_summarizer.llm_client")


class LLMRequestError(RuntimeError):
    """Raised when the LLM server cannot be reached or errors after all retries."""


class LLMClient(ABC):
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        ...

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float) -> str:
        ...


class LlamaCppClient(LLMClient):
    """Client for a running `llama-server` (llama.cpp's OpenAI-compatible server)."""

    def __init__(
        self,
        server_url: str,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.5,
        request_timeout_seconds: float = 120.0,
    ):
        self.server_url = server_url.rstrip("/")
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.request_timeout_seconds = request_timeout_seconds

    def _post_with_retries(self, path: str, payload: dict) -> dict:
        url = f"{self.server_url}{path}"
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(url, json=payload, timeout=self.request_timeout_seconds)
                resp.raise_for_status()
                return resp.json()
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
                last_error = exc
                wait = self.retry_backoff_seconds * attempt
                logger.warning(
                    "request to %s failed (attempt %d/%d): %s -- retrying in %.1fs",
                    url, attempt, self.max_retries, exc, wait,
                )
                time.sleep(wait)
        raise LLMRequestError(
            f"Could not reach llama.cpp server at {url} after {self.max_retries} attempts. "
            f"Is `llama-server` running and reachable at {self.server_url}? "
            f"Last error: {last_error}"
        )

    def count_tokens(self, text: str) -> int:
        data = self._post_with_retries("/tokenize", {"content": text})
        tokens = data.get("tokens")
        if tokens is None:
            raise LLMRequestError(f"/tokenize response missing 'tokens' field: {data}")
        return len(tokens)

    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float) -> str:
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        data = self._post_with_retries("/v1/chat/completions", payload)
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as exc:
            raise LLMRequestError(f"Unexpected response shape from server: {data}") from exc


class MockLLMClient(LLMClient):
    """No-network stand-in for the demo and for tests.

    Approximates llama.cpp's tokenizer with a simple ~4-chars/token heuristic
    (good enough to exercise budget-driven chunking) and "summarizes" by
    extracting a few substantive lines from the input, tagged with the role it
    was called for. This is NOT a real summarizer -- it exists purely to prove
    the chunking / recursive-reduce / tree-walk wiring is correct end to end
    without needing a GPU or a running server.
    """

    def __init__(self, chars_per_token: float = 4.0):
        self.chars_per_token = chars_per_token
        self._call_count = 0

    def count_tokens(self, text: str) -> int:
        return max(1, int(len(text) / self.chars_per_token))

    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float) -> str:
        self._call_count += 1
        if "Mapper" in system_prompt:
            role = "mapper"
        elif "Reducer" in system_prompt:
            role = "reducer"
        elif "Synthesizer" in system_prompt:
            role = "synthesizer"
        elif "Verifier" in system_prompt:
            role = "verifier"
        else:
            role = "unknown"

        lines = [ln.strip("-* \t") for ln in user_prompt.splitlines() if ln.strip()]
        picked = [ln for ln in lines if len(ln) > 8][:3]
        if not picked:
            picked = ["(no extractable content)"]
        digest = "; ".join(picked)[:220]
        result = f"[{role}#{self._call_count}] {digest}"
        logger.debug("mock complete() call #%d role=%s -> %r", self._call_count, role, result[:80])
        return result
