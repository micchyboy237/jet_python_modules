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


def demo_multi_turn_chat():
    """Multi-turn conversation with context."""
    print("=== Multi-turn Chat ===")
    llm = LlamacppLLM()

    messages: list[ChatMessage] = [
        {"role": "system", "content": "You are a helpful math tutor."},
        {"role": "user", "content": "What is the Pythagorean theorem?"},
    ]

    # First turn
    response1 = llm.chat(messages, temperature=0.3)
    print(f"Assistant: {response1}\n")

    # Add response to history
    messages.append({"role": "assistant", "content": response1})

    # Second turn
    messages.append({"role": "user", "content": "Give me an example with numbers."})
    response2 = llm.chat(messages, temperature=0.3)
    print(f"Assistant: {response2}\n")


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

    response = llm.chat(
        messages,
        temperature=0.0,
        enable_thinking=True,
    )
    print(f"Response: {response}\n")


if __name__ == "__main__":
    # Uncomment the demos you want to run
    demo_stream_chat()
    # demo_multi_turn_chat()
    # demo_with_thinking()
