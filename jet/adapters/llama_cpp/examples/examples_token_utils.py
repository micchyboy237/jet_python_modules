from jet.adapters.llama_cpp.token_utils import (
    count_chat_tokens,
    count_tokens,
    detokenize,
    get_detokenizer_fn,
    get_tokenizer,
    get_tokenizer_fn,
    tokenize,
)


def run_demo():
    print("=== Testing get_tokenizer ===")
    try:
        tokenizer = get_tokenizer("llama-3.2:3b", verbose=True)
        print(f"Tokenizer loaded: {type(tokenizer).__name__}")
    except Exception as e:
        print(f"Local tokenizer not available: {e}")

    print("\n=== Testing get_tokenizer_fn ===")
    try:
        encode_fn = get_tokenizer_fn("llama-3.2:3b")
        result = encode_fn("Hello world!")
        print(f"Tokenized: {result[:5]}...")
    except Exception as e:
        print(f"get_tokenizer_fn failed: {e}")

    print("\n=== Testing get_detokenizer_fn ===")
    try:
        decode_fn = get_detokenizer_fn("llama-3.2:3b")
        result = decode_fn([123, 456, 789])
        print(f"Detokenized: {result}")
    except Exception as e:
        print(f"get_detokenizer_fn failed: {e}")

    print("\n=== Local tokenization (default) ===")
    try:
        local_tokens = tokenize("Hello world!", with_pieces=True)
        print("Local tokens:", local_tokens["tokens"][:5], "...")
        local_text = detokenize([123, 456, 789])
        print("Local detokenized:", local_text["content"])
        local_count = count_tokens("This is a test prompt.")
        print("Local token count:", local_count)
    except Exception as e:
        print(f"Local operations failed: {e}")

    print("\n=== Server tokenization (use_server=True) ===")
    try:
        server_tokens = tokenize(
            "Hello world!", add_special=True, with_pieces=True, use_server=True
        )
        print("Server tokens:", server_tokens["tokens"][:5], "...")
        server_text = detokenize([123, 456, 789], use_server=True)
        print("Server detokenized:", server_text["content"])
        server_count = count_tokens("This is a test prompt.", use_server=True)
        print("Server token count:", server_count)
    except Exception as e:
        print(f"Server operations failed (is server running?): {e}")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]

    messages_with_tools = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather like in Paris?"},
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The unit of temperature",
                        },
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    print("\n=== Local chat tokens (default) ===")
    try:
        chat_result_local = count_chat_tokens(messages)
        print(f"Local chat tokens (without tools): {chat_result_local['input_tokens']}")
        chat_result_local_with_tools = count_chat_tokens(
            messages_with_tools, tools=tools
        )
        print(
            f"Local chat tokens (with tools): {chat_result_local_with_tools['input_tokens']}"
        )
    except Exception as e:
        print(f"Local chat count failed: {e}")

    print("\n=== Server chat tokens (use_server=True) ===")
    try:
        chat_result_server = count_chat_tokens(messages, use_server=True)
        print(
            f"Server chat tokens (without tools): {chat_result_server['input_tokens']}"
        )
        chat_result_server_with_tools = count_chat_tokens(
            messages_with_tools, tools=tools, use_server=True
        )
        print(
            f"Server chat tokens (with tools): {chat_result_server_with_tools['input_tokens']}"
        )
    except Exception as e:
        print(f"Server chat count failed: {e}")

    print("\n=== Testing count_tokens with auto-detection ===")
    result = count_tokens("Hello, how are you?")
    print(f"String input (local): {result} tokens")
    result = count_tokens("Hello, how are you?", use_server=True)
    print(f"String input (server): {result} tokens")
    result = count_tokens(messages)
    print(f"Message dicts without tools (local): {result} tokens")
    result = count_tokens(messages_with_tools, tools=tools)
    print(f"Message dicts with tools (local): {result} tokens")
    result = count_tokens(messages, use_server=True)
    print(f"Message dicts without tools (server): {result} tokens")
    result = count_tokens(messages_with_tools, tools=tools, use_server=True)
    print(f"Message dicts with tools (server): {result} tokens")
    result = count_tokens([123, 456, 789])
    print(f"Token list input: {result} tokens")

    print("\n=== Testing list of strings ===")
    string_list = ["Hello world!", "How are you?", "I am doing great today!"]
    result = count_tokens(string_list)
    print(f"List of strings (local, total): {result} tokens")
    result = count_tokens(string_list, prevent_total=True)
    print(f"List of strings (local, individual): {result} tokens")
    result = count_tokens(string_list, use_server=True)
    print(f"List of strings (server, total): {result} tokens")

    print("\n=== Testing empty inputs ===")
    result = count_tokens("")
    print(f"Empty string: {result} tokens")
    result = count_tokens([])
    print(f"Empty list: {result} tokens")
    result = count_tokens([], prevent_total=True)
    print(f"Empty list (individual): {result} tokens")


if __name__ == "__main__":
    run_demo()
