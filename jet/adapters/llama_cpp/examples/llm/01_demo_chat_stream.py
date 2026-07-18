from jet.adapters.llama_cpp.llm import ChatMessage, LlamacppLLM


def demo_stream_chat():
    """Streaming chat example."""
    print("=== Streaming Chat ===")
    llm = LlamacppLLM()

    messages: list[ChatMessage] = [
        {"role": "user", "content": "Count from 1 to 10 slowly."}
    ]

    print("Streaming response:", end=" ")
    response = ""
    for chunk in llm.chat(
        messages,
        stream=True,
        temperature=0.5,
        top_p=0.95,
        presence_penalty=1.5,
        top_k=20,
    ):
        # print(chunk, end="", flush=True)
        response += chunk
    print(f"\nResponse: {response}")
    print("\n")


def demo_with_thinking():
    """Chat with thinking mode enabled."""
    print("=== Chat with Thinking Mode ===")
    llm = LlamacppLLM()

    messages: list[ChatMessage] = [
        {
            "role": "user",
            "content": "Solve this logic puzzle: If all A are B, and some B are C, are all A necessarily C?",
        }
    ]

    response = ""
    for chunk in llm.chat(
        messages,
        stream=True,
        temperature=0.0,
        enable_thinking=True,
    ):
        # print(chunk, end="", flush=True)
        response += chunk
    print(f"Response: {response}\n")
    print("\n")


if __name__ == "__main__":
    # Uncomment the demos you want to run
    demo_stream_chat()
    demo_with_thinking()
