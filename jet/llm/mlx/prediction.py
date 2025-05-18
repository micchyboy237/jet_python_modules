from jet.llm.mlx.mlx_types import LLMModelType
from jet.llm.mlx.models import resolve_model
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import random


def predict_next_word(sentence, top_n=5, model_name: LLMModelType = "llama-3.2-3b-instruct-4bit"):
    """
    Predict the top N most likely next words for an unfinished sentence.

    Args:
        sentence (str): The unfinished sentence.
        top_n (int): Number of top predictions to return (default: 5).
        model_name (str): Name of the MLX model to use.

    Returns:
        dict: Contains predicted words, their probabilities, and additional info.
    """
    # Load model and tokenizer
    model, tokenizer = load(resolve_model(model_name))

    # Prepare prompt
    prompt = sentence
    # if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
    #     messages = [
    #         {"role": "user", "content": f"Predict the next word: {prompt}"}]
    #     prompt = tokenizer.apply_chat_template(
    #         messages, tokenize=False, add_generation_prompt=True)

    # Tokenize input
    tokens = tokenizer.encode(prompt)
    tokens = mx.array([tokens])  # Add batch dimension

    # Get logits
    logits = model(tokens)  # Shape: (batch_size, sequence_length, vocab_size)
    last_logits = logits[0, -1, :]  # Logits for the next token

    # Compute probabilities using softmax
    probs = mx.softmax(last_logits).astype(mx.float32)

    # Get top N token IDs and their probabilities
    top_n_indices = mx.argsort(-probs)[:top_n]  # Sort descending
    top_n_probs = probs[top_n_indices]

    # Decode tokens to words
    top_n_words = [tokenizer.decode([idx.item()]) for idx in top_n_indices]

    # Convert probabilities to percentages
    top_n_percentages = [prob.item() * 100 for prob in top_n_probs]

    # Prepare results
    results = {
        "input_sentence": sentence,
        "top_predictions": [
            {"word": word, "probability": f"{prob:.2f}%"}
            for word, prob in zip(top_n_words, top_n_percentages)
        ],
        "model": model_name,
        "prompt": prompt
    }

    return results


def predict_finishing_words(sentence, top_n=5, model_name: LLMModelType = "llama-3.2-3b-instruct-4bit"):
    """
    Predict the top N most likely finishing words for an unfinished sentence.

    Args:
        sentence (str): The unfinished sentence.
        top_n (int): Number of top finishing words to return (default: 5).
        model_name (str): Name of the MLX model to use.

    Returns:
        dict: Contains predicted finishing words, their probabilities, and additional info.
    """
    # Load model and tokenizer
    model, tokenizer = load(resolve_model(model_name))

    # Prepare prompt
    prompt = sentence
    # if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
    #     messages = [
    #         {"role": "user", "content": f"Complete the sentence: {prompt}"}]
    #     prompt = tokenizer.apply_chat_template(
    #         messages, tokenize=False, add_generation_prompt=True)

    # Tokenize input
    tokens = tokenizer.encode(prompt)
    tokens = mx.array([tokens])  # Add batch dimension

    # Get logits for the next token
    logits = model(tokens)  # Shape: (batch_size, sequence_length, vocab_size)
    last_logits = logits[0, -1, :]  # Logits for the next token

    # Compute probabilities using softmax
    probs = mx.softmax(last_logits).astype(mx.float32)

    # Get top N token IDs and their probabilities
    top_n_indices = mx.argsort(-probs)[:top_n]  # Sort descending
    top_n_probs = probs[top_n_indices]

    # Decode tokens to words
    top_n_words = [tokenizer.decode([idx.item()]) for idx in top_n_indices]

    # Convert probabilities to percentages
    top_n_percentages = [prob.item() * 100 for prob in top_n_probs]

    # Generate full continuation for context (optional, for debugging)
    full_completion = generate(
        model, tokenizer, prompt=prompt, max_tokens=50, verbose=False)

    # Prepare results
    results = {
        "input_sentence": sentence,
        "top_finishing_words": [
            {"word": word, "probability": f"{prob:.2f}%"}
            for word, prob in zip(top_n_words, top_n_percentages)
        ],
        "model": model_name,
        "prompt": prompt,
        "sample_completion": full_completion.strip()
    }

    return results


def predict_top_completions(sentence, top_n=5, model_name: LLMModelType = "llama-3.2-3b-instruct-4bit"):
    """
    Predict the top N most likely sentence completions for an unfinished sentence.

    Args:
        sentence (str): The unfinished sentence.
        top_n (int): Number of top completions to return (default: 5).
        model_name (str): Name of the MLX model to use.

    Returns:
        dict: Contains predicted completions, their relative probabilities, and additional info.
    """
    # Load model and tokenizer
    model, tokenizer = load(resolve_model(model_name))

    # Prepare prompt
    prompt = sentence
    # if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
    #     messages = [
    #         {"role": "user", "content": f"Complete the sentence: {prompt}"}]
    #     prompt = tokenizer.apply_chat_template(
    #         messages, tokenize=False, add_generation_prompt=True)

    # Generate multiple completions with varying parameters
    completions = []
    max_attempts = top_n * 3  # Try up to 3x top_n attempts to get unique completions
    temperatures = [0.6, 0.8, 1.0, 1.2, 1.5]  # Wider range for diversity
    top_k_values = [50, 100, 200, 500]  # Larger top_k for more variety
    attempt = 0

    while len(set(completions)) < top_n and attempt < max_attempts:
        # Select parameters for this attempt
        temp = temperatures[attempt % len(temperatures)]
        top_k = top_k_values[attempt % len(top_k_values)]
        # Random seed for additional diversity
        seed = random.randint(0, 1000000)
        mx.random.seed(seed)

        # Create sampler
        sampler = make_sampler(
            temp=temp,
            top_p=0.95,  # Increased for more diversity
            min_p=0.0,
            min_tokens_to_keep=1,
            top_k=top_k,
            xtc_probability=0.0,
            xtc_threshold=0.1,
            xtc_special_tokens=tokenizer.encode(
                "\n") + list(tokenizer.eos_token_ids),
        )
        # Generate completion
        completion = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=50,
            sampler=sampler,
            verbose=False
        ).strip()
        completions.append(completion)
        attempt += 1

    # Remove duplicates and take top_n
    unique_completions = list(dict.fromkeys(completions))[:top_n]

    # Compute log probabilities for each completion
    completion_log_probs = []
    for completion in unique_completions:
        # Tokenize the full completion
        full_text = prompt + completion
        tokens = tokenizer.encode(full_text)
        tokens = mx.array([tokens])

        # Get logits for the entire sequence
        logits = model(tokens)  # Shape: (1, seq_len, vocab_size)

        # Compute log probabilities for the generated tokens
        log_probs = mx.log(mx.softmax(logits, axis=-1))
        input_len = len(tokenizer.encode(prompt))
        generated_tokens = tokens[0, input_len:]
        generated_log_probs = [
            log_probs[0, input_len + i, token.item()]
            for i, token in enumerate(generated_tokens)
        ]
        # Sum log probabilities
        total_log_prob = sum([lp.item() for lp in generated_log_probs])

        completion_log_probs.append((completion, total_log_prob))

    # Normalize log probabilities to compute relative probabilities
    if completion_log_probs:
        max_log_prob = max(lp for _, lp in completion_log_probs)
        completion_probs = [
            (comp, (lp - max_log_prob)) for comp, lp in completion_log_probs
        ]
        # Convert to relative probabilities (softmax-like normalization)
        exp_scores = [mx.exp(lp).item() for _, lp in completion_probs]
        total_score = sum(exp_scores)
        if total_score > 0:
            relative_probs = [score / total_score *
                              100 for score in exp_scores]
        else:
            # Uniform if all zero
            relative_probs = [
                100.0 / len(completion_probs)] * len(completion_probs)
    else:
        unique_completions = unique_completions or ["No completions generated"]
        relative_probs = [
            100.0 / len(unique_completions)] * len(unique_completions)

    # Sort by probability and ensure top_n results
    sorted_completions = sorted(
        zip(unique_completions, relative_probs),
        key=lambda x: x[1],
        reverse=True
    )

    # Pad with placeholder if fewer than top_n
    while len(sorted_completions) < top_n:
        sorted_completions.append(
            ("Insufficient unique completions generated", 0.0))

    # Prepare results
    results = {
        "input_sentence": sentence,
        "top_completions": [
            {"completion": comp, "probability": f"{prob:.2f}%"}
            for comp, prob in sorted_completions[:top_n]
        ],
        "model": model_name,
        "prompt": prompt,
        "unique_completions_found": len(unique_completions)
    }

    return results
