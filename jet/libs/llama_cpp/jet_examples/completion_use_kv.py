"""
completion_use_kv.py
====================
Demonstrates KV cache reuse via /completion (raw) for JA→EN subtitle translation.

HOW KV CACHE WORKS HERE (vs chat)
-----------------------------------
  - We build the raw prompt string ourselves using the same Jinja template
    structure the server would produce for chat.
  - Because WE control the string, the prefix is guaranteed byte-identical
    on every call — no template re-render variance.
  - This gives the most reliable prefix match and highest cache hit rate.
  - tokens_cached in the response confirms how many tokens were skipped.

RAW PROMPT STRUCTURE (mirrors ministral-3b-instruct.jinja)
-----------------------------------------------------------
  <s>[SYSTEM_PROMPT]{system}[/SYSTEM_PROMPT]
  [INST]{user}
  [IMPORTANT]
  Return ONLY the final English translation.
  ...
  [/IMPORTANT][/INST]

FIXED PREFIX = everything up to and including [INST]  (gets cached)
SUFFIX       = the subtitle text + [IMPORTANT]...[/INST]  (changes each call)
"""

import os
import time

import requests

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL = os.getenv("LLAMA_CPP_LLM_URL", "http://localhost:8080/v1")
COMPLETION_URL = f"{BASE_URL}/completion"

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

# The instruction footer appended after every subtitle line.
# Keeping this identical across calls means it stays in the cached prefix once
# the server has seen it.
INST_FOOTER = (
    "\n\n[IMPORTANT]\n"
    "Return ONLY the final English translation.\n"
    "If there is no spoken content, return an empty response.\n"
    "Do not infer missing dialogue from silence, noise, music, breathing, "
    "or punctuation-only input.\n"
    "No prefixes, no explanations, no extra text.\n"
    "[/IMPORTANT]"
)

SUBTITLE_LINES = [
    "今日はとても疲れた。",
    "あなたのことが好きだよ。",
    "早く行かないと電車に乗り遅れる！",
    "このラーメンは最高に美味しいな。",
    "彼女は昨日から姿を消してしまった。",
]

# ── Prompt builder ────────────────────────────────────────────────────────────


def build_prompt(subtitle: str, prev_pairs: list[tuple[str, str]] = None) -> str:
    """
    Construct a raw prompt string that mirrors the Jinja template output.

    The FIXED PREFIX is everything before the new subtitle.  It is always
    identical across calls so the KV cache can match it.

    Parameters
    ----------
    subtitle      : The new Japanese line to translate.
    prev_pairs    : List of (ja, en) tuples from earlier turns.
                    If provided, prior turns are embedded in the prefix so the
                    model has context — and those tokens are also cached.

    Returns
    -------
    Full raw prompt string ready to send to /completion.
    """
    # BOS token + system prompt (FIXED — always cached after call 1)
    prompt = f"<s>[SYSTEM_PROMPT]{SYSTEM_PROMPT}[/SYSTEM_PROMPT]"

    # Optional prior turns (also fixed once cached)
    if prev_pairs:
        for ja, en in prev_pairs:
            prompt += f"[INST]{ja}{INST_FOOTER}[/INST]{en}</s>"

    # New subtitle — this is the only CHANGING part (suffix)
    prompt += f"[INST]{subtitle}{INST_FOOTER}[/INST]"

    return prompt


def split_prompt_info(prompt: str, subtitle: str) -> tuple[int, int]:
    """
    Estimate prefix length (chars) and suffix length (chars) for display.
    The suffix starts at the last occurrence of the subtitle in the prompt.
    """
    idx = prompt.rfind(subtitle)
    if idx == -1:
        return len(prompt), 0
    return idx, len(prompt) - idx


# ── Core translate function ───────────────────────────────────────────────────


def translate(
    subtitle: str,
    prev_pairs: list[tuple[str, str]] = None,
    slot_id: int = 0,
) -> dict:
    """
    Translate a single subtitle line using /completion.

    Parameters
    ----------
    subtitle   : Japanese text to translate.
    prev_pairs : Prior (ja, en) turn pairs for context.
    slot_id    : KV slot to pin.  0 = first slot.  -1 = auto.

    Returns
    -------
    dict with: translation, slot_id, tokens_cached, tokens_evaluated,
               prompt_chars, latency_ms
    """
    prompt = build_prompt(subtitle, prev_pairs or [])
    prefix_chars, suffix_chars = split_prompt_info(prompt, subtitle)

    payload = {
        "prompt": prompt,
        "n_predict": 128,
        "temperature": 0.0,
        "stop": ["</s>", "[INST]"],  # stop at next turn boundary
        "cache_prompt": True,  # ← ENABLE KV CACHE REUSE
        "id_slot": slot_id,  # ← PIN TO SLOT
        "stream": False,
    }

    t0 = time.perf_counter()
    resp = requests.post(COMPLETION_URL, json=payload, timeout=60)
    resp.raise_for_status()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    data = resp.json()
    translation = data.get("content", "").strip()
    tokens_cached = data.get("tokens_cached", 0)  # llama-server extension
    tokens_evaluated = data.get("tokens_evaluated", 0)
    returned_slot = data.get("id_slot", slot_id)

    return {
        "translation": translation,
        "slot_id": returned_slot,
        "tokens_cached": tokens_cached,
        "tokens_evaluated": tokens_evaluated,
        "prompt_chars": len(prompt),
        "prefix_chars": prefix_chars,
        "suffix_chars": suffix_chars,
        "latency_ms": elapsed_ms,
    }


def print_result(req_num: int, subtitle: str, result: dict) -> None:
    cached = result["tokens_cached"]
    evaluated = result["tokens_evaluated"]
    hit_pct = (cached / evaluated * 100) if evaluated else 0
    cache_tag = "CACHE HIT ✓" if cached > 0 else "cache miss"
    print(
        f"\n[Request {req_num}] {cache_tag}"
        f"\n  JA       : {subtitle}"
        f"\n  EN       : {result['translation']}"
        f"\n  Slot     : {result['slot_id']}"
        f"\n  Prompt   : {result['prompt_chars']} chars  "
        f"(prefix {result['prefix_chars']} / suffix {result['suffix_chars']})"
        f"\n  Tokens   : {evaluated} evaluated  /  {cached} cached ({hit_pct:.0f}%)"
        f"\n  Latency  : {result['latency_ms']:.0f} ms"
    )


# ── Demo: scenario A — stateless (no prior context) ───────────────────────────


def demo_stateless() -> None:
    """
    Sends [SYSTEM_PROMPT]...[INST]{subtitle}[/INST] each time.
    Prefix = system block (identical every call → cached after req 1).
    Suffix = subtitle only.
    """
    print("\n" + "=" * 60)
    print("SCENARIO A — Stateless (no history)")
    print("  Prefix = system block  →  cached after request 1.")
    print("=" * 60)

    for i, subtitle in enumerate(SUBTITLE_LINES, start=1):
        result = translate(subtitle, prev_pairs=[], slot_id=0)
        print_result(i, subtitle, result)


# ── Demo: scenario B — stateful (rolling prior turns in prefix) ───────────────


def demo_stateful() -> None:
    """
    Sends [SYSTEM_PROMPT]...[prior turns][INST]{subtitle}[/INST].
    Prefix = system + all prior turns  →  grows and stays cached.
    Suffix = current subtitle only.
    Each new call has a slightly longer prefix but it extends the cached
    version, so only the DELTA (new prior turn) needs re-processing.
    """
    print("\n" + "=" * 60)
    print("SCENARIO B — Stateful (rolling prior turns in prefix)")
    print("  Prefix grows each call but extends the cached context.")
    print("=" * 60)

    prev_pairs: list[tuple[str, str]] = []
    for i, subtitle in enumerate(SUBTITLE_LINES, start=1):
        result = translate(subtitle, prev_pairs=prev_pairs, slot_id=0)
        print_result(i, subtitle, result)
        prev_pairs.append((subtitle, result["translation"]))


# ── Demo: scenario C — slot isolation (two independent streams) ───────────────


def demo_dual_slot() -> None:
    """
    Uses slot 0 and slot 1 for two separate subtitle streams.
    Each slot maintains its own independent KV cache.
    Useful when translating two speakers or two video feeds in parallel.
    """
    print("\n" + "=" * 60)
    print("SCENARIO C — Dual slot (two independent streams)")
    print("  Slot 0 = stream A,  slot 1 = stream B.")
    print("=" * 60)

    stream_a = SUBTITLE_LINES[:3]
    stream_b = ["こんにちは！", "お腹が空いた。", "もう帰ります。"]

    for i, (sa, sb) in enumerate(zip(stream_a, stream_b), start=1):
        ra = translate(sa, slot_id=0)
        rb = translate(sb, slot_id=1)
        print(f"\n[Round {i}]")
        print(f"  Slot 0  JA: {sa}")
        print(
            f"  Slot 0  EN: {ra['translation']}  "
            f"(cached {ra['tokens_cached']}/{ra['tokens_evaluated']} tokens, "
            f"{ra['latency_ms']:.0f} ms)"
        )
        print(f"  Slot 1  JA: {sb}")
        print(
            f"  Slot 1  EN: {rb['translation']}  "
            f"(cached {rb['tokens_cached']}/{rb['tokens_evaluated']} tokens, "
            f"{rb['latency_ms']:.0f} ms)"
        )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("llama-server KV cache demo — /completion")
    print(f"Server: {BASE_URL}")
    print("\n[Note] tokens_cached > 0 confirms prefix was reused.")
    print("       Latency should drop noticeably after request 1 in each demo.")

    demo_stateless()
    demo_stateful()
    demo_dual_slot()

    print("\nDone.  Compare latency between request 1 (cold) and 2+ (warm).")
