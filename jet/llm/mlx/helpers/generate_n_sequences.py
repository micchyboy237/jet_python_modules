from jet.logger import logger

from jet.transformers.formatters import format_json


def generate_n_sequences(
    prompt: str,
    model: str,
    n: int = 5,
    seed: int = 42,
):
    """
    Generates n sequences with repetition detection.
    """
    from jet.llm.mlx.base import MLX
    import math

    logger.info(f"\nInitializing MLX...")
    mlx = MLX(model, seed=seed)

    logger.info(f"\nGenerating top {n} next tokens:")
    top_n_result = mlx.generate(
        prompt, model, max_tokens=1, logprobs=n, verbose=True)

    top_n_result_choice = top_n_result["choices"][0]
    top_n_logprobs_dict = top_n_result_choice["logprobs"]

    top_n_output_logprobs = top_n_logprobs_dict["top_logprobs"][-1]

    logger.info(f"\nGenerating top {n} next sequences:")
    sequences = []
    for num, (token, logprob) in enumerate(list(top_n_output_logprobs.items()), 1):
        token_string = mlx.tokenizer.decode(token)
        sequence = prompt + token_string
        stream_response = mlx.stream_generate(
            sequence, model, max_tokens=20, verbose=True)

        logger.log(f"\nToken {num}:", f"'{token_string}'",
                   colors=["GRAY", "ORANGE"])

        response = ""
        current_sequence = sequence + " "
        full_sequence = None
        for chunk in stream_response:
            response = chunk["choices"][0]["text"]

            current_sequence += response

            if chunk["choices"][0]["finish_reason"]:
                full_sequence = current_sequence

        sequences.append({
            "text": token_string,
            "token": token,
            "prob": int(math.exp(logprob) * 10000) / 100,
            "logprob": logprob,
            "sequence": full_sequence or current_sequence,
        })

    return sequences
