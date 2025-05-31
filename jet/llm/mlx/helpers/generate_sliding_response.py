from jet.llm.mlx.generation import stream_chat
from jet.llm.mlx.mlx_types import LLMModelType
from jet.llm.mlx.models import resolve_model
from jet.llm.mlx.token_utils import get_tokenizer
import json
import uuid


def count_tokens(text, tokenizer):
    return len(tokenizer.encode(text))


def trim_context(messages, max_context_tokens, tokenizer, preserve_system=True):
    total_tokens = sum(count_tokens(json.dumps(msg), tokenizer)
                       for msg in messages)
    if total_tokens <= max_context_tokens:
        return messages, total_tokens

    trimmed_messages = [messages[0]] if preserve_system and messages else []
    current_tokens = count_tokens(json.dumps(
        messages[0]), tokenizer) if preserve_system and messages else 0

    for msg in messages[1:] if preserve_system else messages:  # Iterate in original order
        msg_tokens = count_tokens(json.dumps(msg), tokenizer)
        if current_tokens + msg_tokens <= max_context_tokens:
            trimmed_messages.append(msg)  # Append to maintain order
            current_tokens += msg_tokens
        else:
            break

    return trimmed_messages, current_tokens


def estimate_remaining_tokens(messages, context_window, tokenizer):
    total_input_tokens = sum(count_tokens(
        json.dumps(msg), tokenizer) for msg in messages)
    return context_window - total_input_tokens


def sliding_window(messages, max_tokens_per_generation, context_window, tokenizer, response_chunk, cutoff_detected):
    if cutoff_detected:
        messages[-1]["content"] = messages[-1]["content"].replace(
            "\n[CONTINUE]", "")
        messages.append(
            {"role": "user", "content": "Continue the previous response where it left off."})

    remaining_tokens = estimate_remaining_tokens(
        messages, context_window, tokenizer)
    if remaining_tokens < 50:
        messages, total_tokens = trim_context(
            messages, context_window - max_tokens_per_generation, tokenizer)
        remaining_tokens = estimate_remaining_tokens(
            messages, context_window, tokenizer)
        print(f"\n[Context trimmed to {total_tokens} tokens]")
    else:
        total_tokens = context_window - remaining_tokens

    current_max_tokens = min(max_tokens_per_generation, remaining_tokens)
    if current_max_tokens <= 0:
        print("Error: No tokens available for generation.")
        return messages, 0, total_tokens

    return messages, current_max_tokens, total_tokens


def generate_sliding_response(messages, max_tokens_per_generation, context_window, model, seed=42):
    tokenizer = get_tokenizer(model)
    full_response = ""
    iteration = 0

    while True:
        iteration += 1
        print(f"\n[Iteration {iteration}]")

        remaining_tokens = estimate_remaining_tokens(
            messages, context_window, tokenizer)
        if remaining_tokens < 50:
            print("Warning: Insufficient tokens for generation. Trimming context.")
            messages, total_tokens = trim_context(
                messages, context_window - max_tokens_per_generation, tokenizer)
            remaining_tokens = estimate_remaining_tokens(
                messages, context_window, tokenizer)

        current_max_tokens = min(max_tokens_per_generation, remaining_tokens)
        if current_max_tokens <= 0:
            print("Error: No tokens available for generation.")
            break

        token_count = 0
        response_chunk = ""
        cutoff_detected = False

        for chunk in stream_chat(
            messages,
            model=model,
            max_tokens=current_max_tokens,
            temperature=0.7,
            top_p=0.9,
            verbose=True,
            seed=seed
        ):
            response_chunk += chunk["choices"][0]["message"]["content"]
            token_count += 1

            if token_count >= current_max_tokens - 50:
                response_chunk += "\n[CONTINUE]"
                cutoff_detected = True
                break

        full_response += response_chunk
        messages.append({"role": "assistant", "content": response_chunk})

        if not cutoff_detected and not response_chunk.endswith("..."):
            break

        messages, current_max_tokens, total_tokens = sliding_window(
            messages, max_tokens_per_generation, context_window, tokenizer, response_chunk, cutoff_detected
        )
        if current_max_tokens <= 0:
            break

    return full_response
