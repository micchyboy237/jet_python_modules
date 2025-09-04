# /Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/mlx/remote/example_usage.py
"""
Example usage of MLX Remote client wrapper functions.

These examples demonstrate:
1. Fetching available models
2. Performing a health check
3. Running a chat completion
4. Streaming a chat completion
5. Running a text generation
6. Streaming a text generation
"""

from jet.llm.mlx.remote import generation as gen
from jet.llm.mlx.chat_history import ChatHistory

DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"


def example_get_models():
    print("=== Available Models ===")
    models = gen.get_models()
    print(models)


def example_health_check():
    print("\n=== Health Check ===")
    status = gen.health_check()
    print(status)


def example_chat():
    print("\n=== Chat Completion ===")
    response = gen.chat(
        "Write a haiku about the ocean.",
        model=DEFAULT_MODEL,
    )
    print(response)


def example_stream_chat():
    print("\n=== Streaming Chat Completion ===")
    for chunk in gen.stream_chat(
        "Explain the benefits of unit testing in Python.",
        model=DEFAULT_MODEL,
    ):
        if "choices" in chunk and chunk["choices"]:
            delta = chunk["choices"][0].get("delta", {}).get("content")
            if delta:
                print(delta, end="", flush=True)
    print("\n--- Stream End ---")


def example_generate():
    print("\n=== Text Generation ===")
    response = gen.generate(
        "Once upon a time in a faraway land,",
        model=DEFAULT_MODEL,
        max_tokens=50,
        temperature=0.7,
    )
    print(response)


def example_stream_generate():
    print("\n=== Streaming Text Generation ===")
    for chunk in gen.stream_generate(
        "In the future, AI assistants will",
        model=DEFAULT_MODEL,
        max_tokens=50,
    ):
        if "choices" in chunk and chunk["choices"]:
            token = chunk["choices"][0].get("text")
            if token:
                print(token, end="", flush=True)
    print("\n--- Stream End ---")


def example_chat_with_history():
    print("\n=== Chat with Conversation History ===")
    history = ChatHistory()
    response1 = gen.chat(
        "Hello, who are you?",
        model=DEFAULT_MODEL,
        with_history=True,
        history=history,
    )
    print("Assistant:", response1)

    response2 = gen.chat(
        "Can you remind me what I just asked?",
        model=DEFAULT_MODEL,
        with_history=True,
        history=history,
    )
    print("Assistant:", response2)


if __name__ == "__main__":
    example_get_models()
    example_health_check()
    example_chat()
    example_stream_chat()
    example_generate()
    example_stream_generate()
    example_chat_with_history()
