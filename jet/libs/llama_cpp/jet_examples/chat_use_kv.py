"""
chat_use_kv.py
==============
Demonstrates KV cache reuse via /v1/chat/completions for JA->EN subtitle translation.

HOW KV CACHE WORKS HERE
------------------------
  - The system prompt is identical on every request.
  - llama-server detects that the beginning (prefix) of the token stream
    hasn't changed and skips re-processing those tokens.
  - Only the new user subtitle line + any new history needs to be computed.
  - We pin requests to slot 0 with id_slot=0 so the server never evicts
    our cached prefix to serve another client.

KEY PARAMS
----------
  cache_prompt: true   — tell the server to look for and reuse prefix matches
  id_slot: 0           — pin to slot 0 so the same KV memory is always reused
  temperature: 0.0     — deterministic output (subtitle translation)
"""

import time

import requests

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL = "http://localhost:8080"
CHAT_URL = f"{BASE_URL}/v1/chat/completions"

# System prompt is intentionally kept static so it gets cached after the
# very first request.  Every subsequent call finds a prefix match.
SYSTEM_PROMPT = (
    "You are a Japanese-to-English subtitle translator.\n\n"
    "Your only task is to translate Japanese subtitle text into natural, "
    "accurate English subtitle text.\n\n"
    "Important rules:\n"
    "1. Output ONLY the final English subtitle text.\n"
    "2. Do NOT add explanations, notes, labels, comments, markdown, quotes, "
    "or any extra text.\n"
    "3. Do NOT say things like 'Translation:' or 'Here is the translation.'\n"
    "4. If input is silence, empty, or non-speech, return an empty response.\n"
    "5. Preserve the original emotional tone and speaker intent exactly."
)

# Sample Japanese subtitle lines to translate
SUBTITLE_LINES = [
    "今日はとても疲れた。",
    "あなたのことが好きだよ。",
    "早く行かないと電車に乗り遅れる！",
    "このラーメンは最高に美味しいな。",
    "彼女は昨日から姿を消してしまった。",
]

# ── Helpers ───────────────────────────────────────────────────────────────────


def translate(
    subtitle: str,
    history: list[dict],
    slot_id: int = 0,
) -> dict:
    """
    Send a single subtitle line to /v1/chat/completions.

    Parameters
    ----------
    subtitle : str
        The Japanese text to translate.
    history : list[dict]
        Previous (user, assistant) message pairs — grows each call so the
        model has context.  The SYSTEM message is always prepended.
    slot_id : int
        KV cache slot to pin to.  Use -1 to let the server pick automatically.

    Returns
    -------
    dict with keys: translation, slot_id, tokens_cached, tokens_evaluated,
                    latency_ms, tokens_generated
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": subtitle})

    payload = {
        "model": "local",  # llama-server ignores this field
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 128,
        "cache_prompt": True,  # ← ENABLE KV CACHE REUSE
        "id_slot": slot_id,  # ← PIN TO A SPECIFIC KV SLOT
        "stream": False,
    }

    t0 = time.perf_counter()
    resp = requests.post(CHAT_URL, json=payload, timeout=60)
    resp.raise_for_status()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    data = resp.json()
    translation = data["choices"][0]["message"]["content"].strip()

    # llama-server returns timing stats in usage (non-standard extension)
    usage = data.get("usage", {})
    tokens_cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    tokens_evaluated = usage.get("prompt_tokens", 0)
    tokens_generated = usage.get("completion_tokens", 0)
    returned_slot = data.get("id_slot", slot_id)

    return {
        "translation": translation,
        "slot_id": returned_slot,
        "tokens_cached": tokens_cached,
        "tokens_evaluated": tokens_evaluated,
        "tokens_generated": tokens_generated,
        "latency_ms": elapsed_ms,
    }


def print_result(req_num: int, subtitle: str, result: dict) -> None:
    cached = result["tokens_cached"]
    evaluated = result["tokens_evaluated"]
    hit_pct = (cached / evaluated * 100) if evaluated else 0
    cache_tag = "CACHE HIT ✓" if cached > 0 else "cache miss"
    gen = result["tokens_generated"]
    print(
        f"\n[Request {req_num}] {cache_tag}"
        f"\n  JA : {subtitle}"
        f"\n  EN : {result['translation']}"
        f"\n  Slot      : {result['slot_id']}"
        f"\n  Tokens    : {evaluated} prompt  /  {cached} cached ({hit_pct:.0f}%)  /  {gen} generated"
        f"\n  Latency   : {result['latency_ms']:.0f} ms"
    )


# ── Demo: scenario A — stateless (no history kept) ────────────────────────────


def demo_stateless() -> list[dict]:
    """
    Each call sends only [system, current_subtitle].
    The system prompt tokens are cached after request 1.
    Good for batch / fire-and-forget translation.
    """
    print("\n" + "=" * 60)
    print("SCENARIO A — Stateless (no history kept)")
    print("  System prompt cached after request 1.")
    print("=" * 60)

    results = []
    for i, subtitle in enumerate(SUBTITLE_LINES, start=1):
        result = translate(subtitle, history=[], slot_id=0)
        print_result(i, subtitle, result)
        results.append(result)
    return results


# ── Demo: scenario B — stateful (rolling history) ─────────────────────────────


def demo_stateful() -> list[dict]:
    """
    Each call appends the previous (user, assistant) pair.
    The growing history gives the model context to resolve ambiguous pronouns
    or repeated speaker patterns across subtitle chunks.
    The system + older history is cached; only the latest turn is new.
    """
    print("\n" + "=" * 60)
    print("SCENARIO B — Stateful (rolling history)")
    print("  System + prior turns cached on each call.")
    print("=" * 60)

    results = []
    history: list[dict] = []
    for i, subtitle in enumerate(SUBTITLE_LINES, start=1):
        result = translate(subtitle, history=history, slot_id=0)
        print_result(i, subtitle, result)
        history.append({"role": "user", "content": subtitle})
        history.append({"role": "assistant", "content": result["translation"]})
        results.append(result)
    return results


def warmup(slot_id: int = 0) -> None:
    """
    Send a silent request to prime the KV cache for the system prompt.
    Uses a short throw-away subtitle so the call is cheap.
    Without this, Scenario A Request 1 is always a cold-cache miss and
    confounds comparison with Scenario B.
    """
    translate("おはよう。", history=[], slot_id=slot_id)


def print_summary(
    stateless: list[dict],
    stateful: list[dict],
) -> None:
    """Print a side-by-side comparison table of both scenarios."""
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Req':<6} {'--- Stateless ---':^38} {'--- Stateful ---':^38}")
    print(
        f"{'':6} {'prompt / cached (%)  gen  lat':^38} {'prompt / cached (%)  gen  lat':^38}"
    )
    print("-" * 84)
    for i, (a, b) in enumerate(zip(stateless, stateful), start=1):

        def fmt(r):
            ev = r["tokens_evaluated"]
            ca = r["tokens_cached"]
            ge = r["tokens_generated"]
            la = int(r["latency_ms"])
            pct = ca / ev * 100 if ev else 0
            return f"{ev:>5}tok / {ca:>3} ({pct:>2.0f}%)  {ge:>2}gen  {la:>5}ms"

        print(f"  R{i}   {fmt(a)}   {fmt(b)}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("llama-server KV cache demo — /v1/chat/completions")
    print(f"Server: {BASE_URL}")

    print("\n[Warming up — priming KV cache slot 0 with the system prompt ...]")
    warmup(slot_id=0)
    print("[Warmup complete — system prompt is now cached]\n")

    a_results = demo_stateless()
    b_results = demo_stateful()
    print_summary(a_results, b_results)

    print("\nDone.  'tokens_cached > 0' confirms a prefix hit.")
    print("Watch 'generated' to understand latency spikes — longer output = slower.")
