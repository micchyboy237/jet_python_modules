import asyncio
import json
import logging
import os
from typing import Any

from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.llama_cpp.factory import get_llm_client
from jet.adapters.llama_cpp.token_utils import count_tokens

from .config import BUDGETS, GRAMMAR_DIR

logger = logging.getLogger("webswarm")


class LocalLLMClient:
    """Budget-aware wrapper using jet.adapters.llama_cpp.factory."""

    def __init__(self):
        self._client = get_llm_client()
        self.tokens_used = 0
        self._grammars: dict[str, str] = {}

    def _load_grammar(self, name: str) -> str:
        if name not in self._grammars:
            path = os.path.join(GRAMMAR_DIR, f"{name}.gbnf")
            if not os.path.isfile(path):
                available = (
                    os.listdir(GRAMMAR_DIR) if os.path.isdir(GRAMMAR_DIR) else []
                )
                raise FileNotFoundError(
                    f"Grammar file not found: {path}\n"
                    f"GRAMMAR_DIR={GRAMMAR_DIR}\n"
                    f"Available files: {available}"
                )
            self._grammars[name] = open(path).read()
            logger.debug(f"Loaded grammar '{name}' from {path}")
        return self._grammars[name]

    async def chat(
        self, messages: list[dict], grammar: str | None = None, max_tokens: int = 512
    ) -> dict | str:
        kwargs: dict[str, Any] = {
            "model": LLM_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "stream": True,
        }

        if grammar:
            grammar_content = self._load_grammar(grammar)
            logger.debug(
                f"Grammar '{grammar}' payload: length={len(grammar_content)}, "
                f"rule_count={grammar_content.count('::=')}, "
                f"enable_thinking=False"
            )
            kwargs["extra_body"] = {
                "grammar": grammar_content,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        else:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}

        loop = asyncio.get_running_loop()

        stream = await loop.run_in_executor(
            None, lambda: self._client.chat.completions.create(**kwargs)
        )

        content_parts: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0

        for chunk in stream:
            if chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens or 0
                completion_tokens = chunk.usage.completion_tokens or 0

            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                content_parts.append(delta.content)
                print(delta.content, end="", flush=True)

        # Newline after stream completes so next log/output starts cleanly
        print()

        self.tokens_used += prompt_tokens + completion_tokens
        content = "".join(content_parts)

        if grammar and not content.strip():
            logger.error(
                f"EMPTY response with grammar '{grammar}'. "
                f"Prompt tokens: {prompt_tokens or 'unknown'}. "
                f"Verify enable_thinking=False is reaching the server."
            )
            return {"error": "EMPTY_RESPONSE", "raw": ""}

        if grammar:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                logger.error(f"Grammar output parse failed: {content[:200]}")
                return {"error": "PARSE_FAIL", "raw": content}

        return content


async def safe_llm_call(
    llm: LocalLLMClient,
    messages: list[dict],
    role: str,
    grammar: str | None = None,
) -> dict | str:
    """Context degradation cascade using exact token counting."""
    budget = BUDGETS[role]
    total = count_tokens(messages)
    limit = sum(budget.values())

    if total <= limit:
        return await llm.chat(messages, grammar=grammar, max_tokens=budget["output"])

    logger.warning(f"[{role}] Context overflow ({total}>{limit}). Trimming docs.")
    user_msgs = [(i, m) for i, m in enumerate(messages) if m["role"] == "user"]
    user_msgs.sort(key=lambda x: len(x[1].get("content", "")), reverse=True)

    if user_msgs:
        idx, msg = user_msgs[0]
        ratio = limit / max(total, 1)
        char_limit = int(len(msg["content"]) * ratio)
        trimmed = msg["content"][:char_limit]
        messages[idx] = {"role": "user", "content": trimmed + "\n[TRUNCATED]"}

    return await llm.chat(messages, grammar=grammar, max_tokens=budget["output"])
