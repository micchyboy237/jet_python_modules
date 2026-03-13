import os
from pathlib import Path

import openai
from jet.adapters.llama_cpp.models import LLAMACPP_MODEL_EMBEDDING_SIZES
from mem0 import Memory

# ──────────────────────────────────────────────
#               CONFIGURATION
# ──────────────────────────────────────────────

DEFAULT_USER_PROMPT = "Continue our conversation naturally. Tell me something interesting or ask me a question."

config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": os.getenv("LLAMA_CPP_LLM_MODEL"),
            "temperature": 0.7,
            "max_tokens": 16000,
            "openai_base_url": os.getenv("LLAMA_CPP_LLM_URL"),
            "api_key": "dummy",
        },
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": os.getenv("LLAMA_CPP_EMBED_MODEL"),
            "embedding_dims": LLAMACPP_MODEL_EMBEDDING_SIZES.get(
                os.getenv("LLAMA_CPP_EMBED_MODEL"), 768
            ),  # fallback 768 if model not in dict
            "openai_base_url": os.getenv("LLAMA_CPP_EMBED_URL"),
            "api_key": "dummy",
        },
    },
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "chat_memories",
            "path": str(
                Path("~/.cache/chroma_db/basic_mem0_chatbot").expanduser().resolve()
            ),
        },
    },
}

# ──────────────────────────────────────────────
#                   MAIN LOOP
# ──────────────────────────────────────────────

memory = Memory.from_config(config)

client = openai.Client(
    base_url=os.getenv("LLAMA_CPP_LLM_URL"),  # ← fixed: was using MODEL instead of URL
    api_key="dummy",
)

user_id = "avb"
messages = []

print("Chat started. Press Enter to use default prompt.")
print(f"Default: {DEFAULT_USER_PROMPT}")
print("Type 'exit' or 'quit' to stop.\n")

while True:
    user_input = input("User: ").strip()

    # Use default when user just presses Enter
    if not user_input:
        user_input = DEFAULT_USER_PROMPT
        print(f"(using default) → {user_input}")

    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye.")
        break

    messages.append({"role": "user", "content": user_input})

    # ── Memory search ────────────────────────────────
    try:
        mem_res = memory.search(
            query=user_input,
            user_id=user_id,
            limit=5,
        )
        related_memories = mem_res.get("results", [])
    except Exception as e:
        print(f"Memory search failed: {e}")
        related_memories = []

    if related_memories:
        related_memories_text = "\n• " + "\n• ".join(
            m["memory"] for m in related_memories
        )
    else:
        related_memories_text = "No relevant memories found yet."

    print("\nRelevant memories:")
    print(related_memories_text)

    system_content = f"""You are a helpful, honest and slightly witty assistant.
Answer naturally and keep the conversation flowing.

Relevant memories from previous interactions:
{related_memories_text}
"""

    full_messages = [{"role": "system", "content": system_content}, *messages]

    # ── Generation ───────────────────────────────────
    try:
        stream = client.chat.completions.create(
            model="llama3.1:latest",  # change if needed
            messages=full_messages,
            stream=True,
            temperature=0.7,
        )

        print("Assistant: ", end="", flush=True)
        answer = ""

        for part in stream:
            if part.choices and part.choices[0].delta.content is not None:
                content = part.choices[0].delta.content
                print(content, end="", flush=True)
                answer += content

        print("\n")

    except Exception as e:
        print(f"\nError during generation: {e}")
        answer = "(generation failed)"

    messages.append({"role": "assistant", "content": answer})

    # ── Save to memory ───────────────────────────────
    try:
        memory.add(
            messages=[
                messages[-2],
                messages[-1],
            ],  # list of dicts: user + assistant turn
            user_id=user_id,
            # metadata={"source": "chat", "timestamp": datetime.now().isoformat()},  # optional
            infer=True,  # let Mem0 extract facts / summarize (recommended)
        )
        print("(memory saved)")

    except Exception as e:
        print(f"Warning: could not save memory → {e}")
