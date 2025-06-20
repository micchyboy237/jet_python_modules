import re
from typing import TypedDict, Optional, Sequence, Iterator
from dataclasses import dataclass
import logging
from jet.llm.mlx.generation import stream_chat, LLMModelType

logger = logging.getLogger(__name__)


@dataclass
class EvaluationComment:
    score: float
    explanation: str
    question: str
    answer: str


@dataclass
class EvaluationDetails:
    comments: list[EvaluationComment]


@dataclass
class EvaluationResult:
    query: Optional[str] = None
    contexts: Optional[list[str]] = None
    score: Optional[float] = None
    feedback: Optional[str] = None
    passing: Optional[bool] = None
    details: Optional[EvaluationDetails] = None
    excerpts: Optional[list[str]] = None


EVAL_QUESTIONS = [
    "Does the retrieved context match the subject matter of the user's query?",
    "Can the retrieved context be used exclusively to provide an answer to the user's query?",
]

# Updated evaluation template with new scoring guide
CONTEXT_EVAL_TEMPLATE = (
    "Your task is to evaluate whether the provided context is relevant to the user's query by answering the following questions:\n"
    "{questions_str}\n"
    "For each question, assign a score based on how well the context addresses it:\n"
    "- 1.0 = Fully and concretely answered with clear, complete info\n"
    "- 0.5 = Partially answered or missing important details\n"
    "- 0.0 = Not answered at all or only vaguely mentioned\n\n"
    "For each question, provide:\n"
    "- Explanation: Short explanation of the score\n"
    "- Score: (0.0, 0.5, or 1.0)\n\n"
    "Follow these examples below:\n"
    "Example 1:\n"
    "Query: \"What is the capital of France?\"\n"
    "<context>\nThe theory of relativity was developed by Albert Einstein.\n</context>\n"
    "Feedback:\n"
    "Q1: Score - 0.0 | Explanation - Context is unrelated to France or its capital.\n"
    "Q2: Score - 0.0 | Explanation - Context cannot answer the query.\n"
    "Example 2:\n"
    "Query: \"What is the capital of France?\"\n"
    "<context>\nParis hosts many tourists in France.\n</context>\n"
    "Feedback:\n"
    "Q1: Score - 0.5 | Explanation - Context mentions Paris and France, partially relevant.\n"
    "Q2: Score - 0.0 | Explanation - Context does not specify Paris as the capital.\n"
    "Example 3:\n"
    "Query: \"What is the capital of France?\"\n"
    "<context>\nThe capital of France is Paris.\n</context>\n"
    "Feedback:\n"
    "Q1: Score - 1.0 | Explanation - Context directly mentions the capital of France.\n"
    "Q2: Score - 1.0 | Explanation - Context fully answers the query.\n"
    "Given the instructions above, generate feedback.\n"
    "Query: \"{query_str}\"\n"
    "<context>\n{context_str}\n</context>\n"
    "Feedback:\n"
)

# Adjusted for new max score of 2.0 (1.0 per question)
_DEFAULT_SCORE_THRESHOLD = 2.0
_PASSING_SCORE_THRESHOLD = 1.0  # Adjusted for new scoring system


def default_parser_function(response: str) -> tuple[Optional[float], Optional[str]]:
    """Parses the response to extract score and feedback."""
    score_match = re.search(r"\[RESULT\] (\d+\.\d+)", response)
    score = float(score_match.group(1)) if score_match else None
    feedback = response if response else None
    return score, feedback


def parse_excerpts(feedback: Optional[str]) -> list[str]:
    """Extracts relevant excerpts from feedback."""
    if not feedback:
        return []
    excerpts = []
    for line in feedback.split("\n"):
        if re.match(r"Q\d+: Score - (0\.0|0\.5|1\.0) \| Explanation - .+", line):
            excerpts.append(line.strip())
    return excerpts


def parse_feedback(feedback: Optional[str]) -> EvaluationDetails:
    """Parses feedback to extract detailed evaluation comments."""
    if not feedback:
        return EvaluationDetails(comments=[])
    comments = []
    for line in feedback.split("\n"):
        match = re.match(
            r"Q(\d+): Score - (0\.0|0\.5|1\.0) \| Explanation - (.+)", line)
        if match:
            question_idx = int(match.group(1)) - 1
            score = float(match.group(2))
            explanation = match.group(3)
            question = EVAL_QUESTIONS[question_idx] if question_idx < len(
                EVAL_QUESTIONS) else ""
            comments.append(EvaluationComment(
                score=score,
                explanation=explanation,
                question=question,
                answer=""  # No YES/NO answer anymore
            ))
    return EvaluationDetails(comments=comments)


def evaluate_context_relevancy(
    query: str,
    contexts: str | list[str],
    model: LLMModelType = "qwen3-1.7b-4bit",
    questions: list[str] = EVAL_QUESTIONS,
    eval_template: str = CONTEXT_EVAL_TEMPLATE,
    score_threshold: float = _DEFAULT_SCORE_THRESHOLD,
    passing_score_threshold: float = _PASSING_SCORE_THRESHOLD,
    **kwargs
) -> EvaluationResult:
    """Evaluates the relevancy of the context to the query using MLX stream_chat."""
    if isinstance(contexts, str):
        contexts = [contexts]

    context_str = "\n".join(contexts)
    questions_str = "\n".join(
        [f"{idx + 1}. {q}" for idx, q in enumerate(questions)])
    prompt = eval_template.format(
        questions_str=questions_str,
        query_str=query,
        context_str=context_str
    )

    logger.debug("Evaluating context relevancy with prompt:\n%s", prompt)

    response_text = ""
    for response in stream_chat(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        max_tokens=512,
        temperature=0.0,
        verbose=True,
        stop=["\n\n"],
        logit_bias=["Q1", "Q2", " 1.0", " 0.5",
                    " 0.0", " Score", " Explanation"],
        **kwargs
    ):
        if response.get("choices"):
            content = response["choices"][0].get(
                "message", {}).get("content", "")
            response_text += content

    details = parse_feedback(response_text)
    score = sum(
        comment.score for comment in details.comments) if details.comments else 0.0
    feedback = response_text if response_text else None
    passing = score >= passing_score_threshold
    excerpts = parse_excerpts(feedback) if passing else []

    return EvaluationResult(
        query=query,
        contexts=contexts,
        score=score,
        feedback=feedback,
        passing=passing,
        details=details,
        excerpts=excerpts
    )
