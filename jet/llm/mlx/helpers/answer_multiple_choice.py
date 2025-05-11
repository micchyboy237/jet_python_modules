from typing import List, Dict, Union, Optional, Callable, TypedDict
from jet.llm.mlx.config import DEFAULT_MODEL
from jet.llm.mlx.mlx_types import ModelType
from jet.llm.mlx.models import resolve_model
from jet.llm.mlx.token_utils import tokenize_strings
from jet.logger import logger
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate, generate_step
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx_lm.utils import TokenizerWrapper


class ModelLoadError(Exception):
    pass


class InvalidMethodError(Exception):
    pass


class PromptFormattingError(Exception):
    pass


class TokenEncodingError(Exception):
    pass


class GenerationError(Exception):
    pass


class InvalidOutputError(Exception):
    pass


class ChatMessage(TypedDict):
    role: str
    content: str


class AnswerResult(TypedDict):
    answer: str
    token_id: int
    is_valid: bool
    method: str
    error: Optional[str]


def answer_multiple_choice(
    question: str,
    choices: List[str],
    model_path: ModelType = DEFAULT_MODEL,
    method: str = "stream_generate",
    max_tokens: int = 1,
    temperature: float = 0.0,
    top_p: float = 0.9
) -> AnswerResult:
    """
    General function for handling multiple-choice questions. It will ask the model to choose one
    of the provided options based on the question asked.
    """
    model_path = resolve_model(model_path)
    try:
        try:
            model, tokenizer = load(model_path)
        except Exception as e:
            raise ModelLoadError(f"Error loading model or tokenizer: {e}")

        if method not in ["stream_generate", "generate_step"]:
            raise InvalidMethodError("Invalid method specified.")

        system_prompt = f"Answer the following question by choosing one of the options provided without any additional text.\nOptions:\n{'\n'.join(choices)}"
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {"role": "user", "content": question}
        ]
        logger.gray("System:")
        logger.debug(system_prompt)
        logger.gray("Tokenized System:")
        logger.debug(tokenize_strings(system_prompt, model_path))
        logger.gray("User:")
        logger.debug(question)
        logger.newline()

        formatted_prompt: str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Encode each option to handle logits processing later
        option_tokens = []
        for choice in choices:
            token = tokenizer.encode(choice, add_special_tokens=False)[0]
            option_tokens.append(token)
            logger.log(f"Token for '{choice}':",
                       token, colors=["GRAY", "ORANGE"])

        logit_bias = {token: 0.0 for token in option_tokens}

        logits_processors = make_logits_processors(logit_bias=logit_bias)
        sampler = make_sampler(temp=temperature, top_p=top_p)

        stop_tokens = tokenizer.encode("\n") + list(tokenizer.eos_token_ids)

        answer = ""
        token_id = -1
        if method == "stream_generate":
            for output in stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=formatted_prompt,
                max_tokens=max_tokens,
                logits_processors=logits_processors,
                sampler=sampler
            ):
                if output.token in stop_tokens:
                    raise InvalidOutputError(
                        "Generated token is a stop token.")
                answer += output.text
                token_id = output.token
                break
        else:
            input_ids = mx.array(tokenizer.encode(
                formatted_prompt, add_special_tokens=False))
            prompt_cache = None
            for token, _ in generate_step(
                model=model,
                prompt=input_ids,
                max_tokens=max_tokens,
                logits_processors=logits_processors,
                sampler=sampler,
                prompt_cache=prompt_cache
            ):
                if token in stop_tokens:
                    raise InvalidOutputError(
                        "Generated token is a stop token.")
                answer = tokenizer.decode([token])
                token_id = token
                break

        # Check that the answer is one of the provided options
        if answer not in choices:
            raise InvalidOutputError(
                f"Output '{answer}' is not one of the provided choices.")

        return AnswerResult(answer=answer, token_id=token_id, is_valid=True, method=method, error=None)

    except Exception as e:
        return AnswerResult(answer="", token_id=-1, is_valid=False, method=method, error=str(e))
