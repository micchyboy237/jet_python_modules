"""
Demo script showing how to use count_tokens with different input types.

This demonstrates:
1. Counting tokens for single strings
2. Counting tokens for lists of strings (with and without prevent_total)
3. Counting tokens for chat messages (list of dicts)
4. Counting tokens per message vs total conversation
"""

from jet.adapters.llama_cpp.token_utils import count_tokens


def demo_single_string():
    """Demo: Count tokens for a single string"""
    print("=" * 60)
    print("Demo 1: Single String")
    print("=" * 60)

    text = "Hello, world! This is a test."
    count = count_tokens(text)

    print(f"Text: '{text}'")
    print(f"Token count: {count}")
    print(f"Return type: {type(count).__name__}")
    print()


def demo_list_of_strings_sum():
    """Demo: Count total tokens for list of strings (default behavior)"""
    print("=" * 60)
    print("Demo 2: List of Strings - Total Count (prevent_total=False)")
    print("=" * 60)

    texts = ["Hello", "World", "How are you?"]
    total_count = count_tokens(texts, prevent_total=False)

    print(f"Texts: {texts}")
    print(f"Total token count: {total_count}")
    print(f"Return type: {type(total_count).__name__}")
    print()


def demo_list_of_strings_per_item():
    """Demo: Count tokens per string in list"""
    print("=" * 60)
    print("Demo 3: List of Strings - Per-Item Counts (prevent_total=True)")
    print("=" * 60)

    texts = ["Hello", "World", "How are you?"]
    counts = count_tokens(texts, prevent_total=True)

    print(f"Texts: {texts}")
    print(f"Per-item token counts: {counts}")
    print(f"Return type: {type(counts).__name__}")

    # Show individual breakdown
    for text, count in zip(texts, counts):
        print(f"  '{text}': {count} tokens")
    print()


def demo_chat_messages_total():
    """Demo: Count total tokens for chat conversation"""
    print("=" * 60)
    print("Demo 4: Chat Messages - Total Count (prevent_total=False)")
    print("=" * 60)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I help you today?"},
        {"role": "user", "content": "What's the weather like?"},
    ]

    total_count = count_tokens(messages, prevent_total=False)

    print(f"Messages: {len(messages)} messages")
    for i, msg in enumerate(messages, 1):
        print(f"  {i}. [{msg['role']}] {msg['content']}")
    print(f"\nTotal token count (with template): {total_count}")
    print(f"Return type: {type(total_count).__name__}")
    print()


def demo_chat_messages_per_message():
    """Demo: Count tokens per message in conversation"""
    print("=" * 60)
    print("Demo 5: Chat Messages - Per-Message Counts (prevent_total=True)")
    print("=" * 60)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I help you today?"},
        {"role": "user", "content": "What's the weather like?"},
    ]

    counts = count_tokens(messages, prevent_total=True)

    print(f"Messages: {len(messages)} messages")
    print(f"Per-message token counts: {counts}")
    print(f"Return type: {type(counts).__name__}")
    print()

    # Show individual breakdown
    for i, (msg, count) in enumerate(zip(messages, counts), 1):
        print(f"  {i}. [{msg['role']}] '{msg['content']}' → {count} tokens")
    print()


def demo_multimodal_content():
    """Demo: Count tokens for messages with multimodal content"""
    print("=" * 60)
    print("Demo 6: Multimodal Content")
    print("=" * 60)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.jpg"},
                },
            ],
        },
        {"role": "assistant", "content": "I see a cat sitting on a windowsill."},
    ]

    # Per-message counts (only counts text parts)
    counts = count_tokens(messages, prevent_total=True)

    print("Message 1 (multimodal):")
    print(f"  Content: {messages[0]['content']}")
    print(f"  Token count (text only): {counts[0]}")

    print("\nMessage 2 (text only):")
    print(f"  Content: '{messages[1]['content']}'")
    print(f"  Token count: {counts[1]}")
    print()


def demo_empty_inputs():
    """Demo: Handle empty inputs"""
    print("=" * 60)
    print("Demo 7: Empty Inputs")
    print("=" * 60)

    # Empty list with prevent_total=True
    empty_list_counts = count_tokens([], prevent_total=True)
    print(f"Empty list (prevent_total=True): {empty_list_counts}")
    print(f"Type: {type(empty_list_counts).__name__}")

    # Empty list with prevent_total=False
    empty_list_total = count_tokens([], prevent_total=False)
    print(f"Empty list (prevent_total=False): {empty_list_total}")
    print(f"Type: {type(empty_list_total).__name__}")
    print()


def demo_comparison():
    """Demo: Compare per-item vs total counting"""
    print("=" * 60)
    print("Demo 8: Comparison - Per-Item vs Total")
    print("=" * 60)

    messages = [
        {"role": "user", "content": "Tell me a joke"},
        {
            "role": "assistant",
            "content": "Why did the chicken cross the road? To get to the other side!",
        },
    ]

    # Per-message counts
    per_message = count_tokens(messages, prevent_total=True)
    per_message_sum = sum(per_message)

    # Total count (uses chat template)
    total_with_template = count_tokens(messages, prevent_total=False)

    print("Messages:")
    for i, msg in enumerate(messages, 1):
        print(f"  {i}. [{msg['role']}] {msg['content']}")

    print(f"\nPer-message counts: {per_message}")
    print(f"Sum of per-message counts: {per_message_sum}")
    print(f"Total with chat template: {total_with_template}")
    print(
        f"\nDifference (template overhead): {total_with_template - per_message_sum} tokens"
    )
    print("(Template adds special tokens, role markers, etc.)")
    print()


def main():
    """Run all demos"""
    print("\n" + "=" * 60)
    print("count_tokens() Demo Script")
    print("=" * 60 + "\n")

    demo_single_string()
    demo_list_of_strings_sum()
    demo_list_of_strings_per_item()
    demo_chat_messages_total()
    demo_chat_messages_per_message()
    demo_multimodal_content()
    demo_empty_inputs()
    demo_comparison()

    print("=" * 60)
    print("All demos completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
