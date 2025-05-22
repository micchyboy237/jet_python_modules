from typing import List, Dict, Optional, TypedDict
from jet.llm.mlx.mlx_types import LLMModelType
from jet.llm.mlx.models import resolve_model
from jet.logger import logger
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx_lm.utils import TokenizerWrapper

mx.random.seed(42)


class ModelLoadError(Exception):
    """Raised when model or tokenizer loading fails."""
    pass


class InvalidMethodError(Exception):
    """Raised when an invalid generation method is specified."""
    pass


class PromptFormattingError(Exception):
    """Raised when prompt formatting fails."""
    pass


class TokenEncodingError(Exception):
    """Raised when token encoding fails."""
    pass


class GenerationError(Exception):
    """Raised when generation fails."""
    pass


class InvalidOutputError(Exception):
    """Raised when the generated answer is not 'Yes' or 'No'."""
    pass


class InvalidInputError(Exception):
    """Raised when questions or contexts are empty or invalid."""
    pass


class ChatMessage(TypedDict):
    role: str
    content: str


class QuestionContext(TypedDict):
    question: str
    context: str


class AnswerResult(TypedDict):
    question: str
    context: str
    answer: str
    token_id: int
    is_valid: bool
    error: Optional[str]


class ModelComponents:
    """Encapsulates model and tokenizer for easier management."""

    def __init__(self, model, tokenizer: TokenizerWrapper):
        self.model = model
        self.tokenizer = tokenizer


def load_model_components(model_path: LLMModelType) -> ModelComponents:
    """Loads model and tokenizer from the specified path."""
    try:
        model, tokenizer = load(resolve_model(model_path))
        return ModelComponents(model, tokenizer)
    except Exception as e:
        raise ModelLoadError(f"Error loading model or tokenizer: {e}")


def validate_method(method: str) -> None:
    """Validates the generation method."""
    if method != "generate_step":
        raise InvalidMethodError(
            "Invalid method specified. Only 'generate_step' is supported.")


def validate_inputs(questions_contexts: List[QuestionContext]) -> None:
    """Validates that questions and contexts are non-empty."""
    if not questions_contexts:
        raise InvalidInputError("Questions and contexts list cannot be empty.")
    for qc in questions_contexts:
        if not qc["question"].strip():
            raise InvalidInputError(f"Question cannot be empty: {qc}")
        if not qc["context"].strip():
            raise InvalidInputError(
                f"Context cannot be empty for question: {qc['question']}")


def create_system_prompt() -> str:
    """Creates a system prompt for yes/no answers with context and few-shot examples."""
    return (
        "You are an assistant that answers questions based only on the given context.\n"
        "Respond strictly with 'Yes' or 'No'.\n"
        "\n"
        "Examples:\n"
        "Context: Venus is the second planet from the Sun and has no natural moons.\n"
        "Question: Does Venus have one or more moons?\n"
        "Answer: No\n"
        "\n"
        "Context: Mars has two small moons named Phobos and Deimos.\n"
        "Question: Does Mars have moons?\n"
        "Answer: Yes\n"
        "\n"
        "Context: Jupiter is the largest planet and has at least 79 known moons.\n"
        "Question: Is Jupiter moonless?\n"
        "Answer: No\n"
        "\n"
        "Context: Saturn has 83 moons with confirmed orbits.\n"
        "Question: Does Saturn have moons?\n"
        "Answer: Yes\n"
        "\n"
        "Given the context, answer the following question with only 'Yes' or 'No'. Ensure accuracy."
    )


def format_chat_messages(
    question: str,
    context: str,
    system_prompt: Optional[str] = None
) -> List[ChatMessage]:
    """Formats the system and user messages for the chat template."""
    user_content = f"Context: {context}\nQuestion: {question}\nAnswer:"
    return [
        {"role": "system", "content": system_prompt or create_system_prompt()},
        {"role": "user", "content": user_content}
    ]


def compute_confidence_scores(
    model,
    input_ids: mx.array,
    choice_token_map: Dict[str, List[int]]
) -> Dict[str, float]:
    """Computes normalized confidence scores from model logits for each choice."""
    try:
        if len(input_ids.shape) == 1:
            input_ids = input_ids[None, :]
        logger.debug(f"Input IDs shape: {input_ids.shape}")
        model_output = model(input_ids)
        if len(model_output.shape) != 3:
            raise ValueError(
                f"Unexpected model output shape: {model_output.shape}")
        logits = model_output[0, -1]
        probs = mx.softmax(logits, axis=-1)
        raw_confidence_scores = {}
        for choice, tokens in choice_token_map.items():
            if tokens:
                token_probs = [float(probs[token_id]) for token_id in tokens]
                raw_confidence_scores[choice] = sum(
                    token_probs) / len(token_probs) if token_probs else 0.0
                logger.debug(
                    f"Choice: {choice}, Token IDs: {tokens}, Raw Prob: {raw_confidence_scores[choice]}")
        total_prob = sum(raw_confidence_scores.values())
        if total_prob == 0:
            logger.warning("Total probability is zero, returning raw scores")
            return raw_confidence_scores
        normalized_confidence_scores = {
            choice: prob / total_prob for choice, prob in raw_confidence_scores.items()
        }
        return normalized_confidence_scores
    except Exception as e:
        logger.error(f"Error computing confidence scores: {str(e)}")
        return {}


def answer_multiple_yes_no_with_context(
    questions_contexts: List[QuestionContext],
    model_path: LLMModelType,
    method: str = "generate_step",
    max_tokens: int = 1,
    temperature: float = 0.1,
    top_p: float = 0.1,
    system_prompt: Optional[str] = None
) -> List[AnswerResult]:
    """Answers multiple yes/no questions with context using the specified model."""
    validate_method(method)
    if max_tokens == 0 or max_tokens < -1:
        raise ValueError("Max tokens can only be -1 or a positive integer.")
    validate_inputs(questions_contexts)

    try:
        model_components = load_model_components(model_path)

        # Encode 'Yes' and 'No' tokens
        try:
            yes_tokens = model_components.tokenizer.encode(
                "Yes", add_special_tokens=False)
            no_tokens = model_components.tokenizer.encode(
                "No", add_special_tokens=False)
            yes_token = yes_tokens[0]
            no_token = no_tokens[0]
            choice_token_map = {"Yes": yes_tokens, "No": no_tokens}
            logger.log("Token for 'Yes':",
                       yes_tokens[0], colors=["GRAY", "ORANGE"])
            logger.log("Token for 'No':",
                       no_tokens[0], colors=["GRAY", "ORANGE"])
        except Exception as e:
            raise TokenEncodingError(f"Error encoding tokens: {e}")

        # Setup generation parameters
        logit_bias = {
            yes_token: 0.0,
            no_token: 0.0,
            **{i: -1e9 for i in range(model_components.tokenizer.vocab_size) if i not in [yes_token, no_token]}
        }
        logits_processors = make_logits_processors(logit_bias=logit_bias)
        sampler = make_sampler(temp=temperature, top_p=top_p)
        stop_tokens = model_components.tokenizer.encode(
            "\n") + list(model_components.tokenizer.eos_token_ids)

        results = []
        for qc in questions_contexts:
            question = qc["question"]
            context = qc["context"]
            try:
                # Format prompt
                messages = format_chat_messages(
                    question, context, system_prompt)
                formatted_prompt = model_components.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                # Generate answer
                answer = ""
                token_id = -1
                input_ids = mx.array(model_components.tokenizer.encode(
                    formatted_prompt, add_special_tokens=False))
                for token, _ in generate_step(
                    model=model_components.model,
                    prompt=input_ids,
                    max_tokens=max_tokens,
                    logits_processors=logits_processors,
                    sampler=sampler,
                    prompt_cache=None
                ):
                    if token in stop_tokens:
                        raise InvalidOutputError(
                            "Generated token is a stop token.")
                    answer = model_components.tokenizer.decode([token]).strip()
                    token_id = token
                    break

                # Compute confidence scores and override if necessary
                confidence_scores = compute_confidence_scores(
                    model_components.model, input_ids, choice_token_map
                )
                if confidence_scores:
                    most_confident_choice = max(
                        confidence_scores, key=confidence_scores.get)
                    logger.debug(
                        f"Normalized confidence scores: {confidence_scores}")
                    logger.debug(
                        f"Most confident choice: {most_confident_choice} ({confidence_scores[most_confident_choice]})")
                    if answer != most_confident_choice:
                        logger.warning(
                            f"Generated answer '{answer}' differs from most confident choice '{most_confident_choice}'. Overriding.")
                        answer = most_confident_choice
                        token_id = choice_token_map[most_confident_choice][
                            0] if choice_token_map[most_confident_choice] else -1

                if answer.lower() not in ["yes", "no"]:
                    raise InvalidOutputError(
                        f"Output '{answer}' is not 'Yes' or 'No' for question: {question}")

                results.append(AnswerResult(
                    question=question,
                    context=context,
                    answer=answer,
                    token_id=token_id,
                    is_valid=True,
                    error=None
                ))
            except Exception as e:
                logger.error(
                    f"Error processing question '{question}': {str(e)}")
                results.append(AnswerResult(
                    question=question,
                    answer="",
                    token_id=-1,
                    is_valid=False,
                    error=str(e)
                ))

        return results

    except (ModelLoadError, TokenEncodingError, InvalidMethodError, InvalidInputError, ValueError) as e:
        raise
    except Exception as e:
        logger.error(f"Error in answer_multiple_yes_no_with_context: {str(e)}")
        raise
