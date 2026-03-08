import os
import sys
import time

from openai import OpenAI

# Connect to local Ollama
client = OpenAI(
    base_url=os.getenv("OLLAMA_LLM_URL"),
    api_key="ollama",  # dummy — ignored
)


def stream_response(messages, model=os.getenv("OLLAMA_LLM_MODEL")):
    print("\n" + "═" * 70, flush=True)
    print(" Model is answering... ".center(70, "═"), flush=True)
    print("═" * 70 + "\n", flush=True)

    start = time.time()

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,  # low → precise reasoning
        max_tokens=1000,
        top_p=0.92,
        stream=True,
    )

    print("→ ", end="", flush=True)

    full_text = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            delta = chunk.choices[0].delta.content
            full_text += delta
            print(delta, end="", flush=True)  # live flush
            sys.stdout.flush()  # force flush in some terminals

    elapsed = time.time() - start
    print("\n\n" + "─" * 70, flush=True)
    print(f"Done in {elapsed:.2f} seconds • model: {model}", flush=True)
    print("─" * 70 + "\n", flush=True)

    return full_text


# ────────────────────────────────────────────────
# SAMPLE INPUT — just run this script
# ────────────────────────────────────────────────

sample_messages = [
    {
        "role": "system",
        "content": "You are an extremely precise, step-by-step reasoning assistant. Especially good at mathematics, logic puzzles, physics, and clear explanations. Always think aloud and verify your steps.",
    },
    {
        "role": "user",
        "content": "A store offers a 20% discount on a $150 item, but adds 8% sales tax on the discounted price. What is the final amount the customer pays? Show every calculation step clearly.",
    },
]

print("Running sample query:\n")
print("User:", sample_messages[-1]["content"], "\n")

stream_response(sample_messages)
