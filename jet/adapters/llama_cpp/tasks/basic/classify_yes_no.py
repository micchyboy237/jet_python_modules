"""Binary yes/no relevance classification via constrained generation.

Equivalent to jet/models/tasks/classify_yes_no.py.
The MLX version extracts raw logits for yes/no tokens from the embedding model.
Since llama.cpp's /v1/embeddings endpoint returns vectors (not logits), this
adapter uses constrained chat generation with logit_bias to force yes/no output,
then maps the result to a probability-like score (1.0 for yes, 0.0 for no).
"""

from typing import Optional, Union

from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.llama_cpp.llm_utils import chat
from jet.adapters.llama_cpp.token_utils import get_tokenizer
from jet.logger import logger

DEFAULT_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


def _build_yes_no_logit_bias(model: str) -> dict[str, int]:
    """Build logit_bias encouraging only 'yes' and 'no' tokens."""
    tokenizer = get_tokenizer(model)
    bias: dict[str, int] = {}
    for label in ["yes", "no"]:
        tokens = tokenizer.encode(label, add_special_tokens=False)
        if tokens:
            bias[str(tokens[0])] = 100
            logger.debug(f"logit_bias: '{label}' -> token {tokens[0]} (bias=100)")
    return bias


def classify_yes_no(
    queries: Union[str, list[str]],
    documents: list[str],
    instruction: Optional[str] = None,
    model: str | None = None,
    temperature: float = 0.1,
) -> list[float]:
    """Classify query-document pairs as yes/no relevance.

    Args:
        queries: Single query or list of queries (one per document).
        documents: List of documents to evaluate.
        instruction: Task instruction. Defaults to web search relevance.
        model: LLM model key. Defaults to LLM_MODEL.
        temperature: Sampling temperature (default: 0.1).

    Returns:
        List of scores where 1.0 = yes (relevant), 0.0 = no (not relevant).
    """
    resolved_model = model or LLM_MODEL
    task_instruction = instruction or DEFAULT_INSTRUCTION

    if isinstance(queries, str):
        queries = [queries]

    if len(queries) != len(documents):
        raise ValueError(
            f"Number of queries ({len(queries)}) must match documents ({len(documents)})"
        )

    logger.info(
        f"classify_yes_no: {len(queries)} pairs, model={resolved_model}, "
        f"instruction='{task_instruction[:60]}...'"
    )

    logit_bias = _build_yes_no_logit_bias(resolved_model)
    scores: list[float] = []

    for i, (query, doc) in enumerate(zip(queries, documents)):
        user_content = (
            f"<Instruct>: {task_instruction}\n<Query>: {query}\n<Document>: {doc}"
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "Judge whether the Document meets the requirements based on "
                    "the Query and the Instruct provided. "
                    'Note that the answer can only be "yes" or "no".'
                ),
            },
            {"role": "user", "content": user_content},
        ]

        try:
            result = chat(
                prompt="",
                model=resolved_model,
                messages=messages,
                max_tokens=1,
                temperature=temperature,
                logit_bias=logit_bias,
                stop=["\n"],
            )
            answer = result.content.strip().lower()
            logger.debug(f"Pair {i}: raw output='{answer}'")

            if answer.startswith("yes"):
                scores.append(1.0)
            elif answer.startswith("no"):
                scores.append(0.0)
            else:
                logger.warning(
                    f"Unexpected output '{answer}' for pair {i}, defaulting to 0.0"
                )
                scores.append(0.0)

        except Exception as e:
            logger.error(f"Classification failed for pair {i}: {e}")
            scores.append(0.0)

    yes_count = sum(1 for s in scores if s > 0.5)
    logger.info(
        f"classify_yes_no complete: {yes_count}/{len(scores)} classified as 'yes'"
    )
    return scores


if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table

    console = Console()

    task = "Given a web search query, retrieve relevant passages that answer the query"
    queries = [
        "What is the capital of China?",
        "Explain gravity",
        "How do I cook pasta?",
    ]
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. "
        "It gives weight to physical objects and is responsible for the movement of planets around the sun.",
        "Machine learning is a subset of artificial intelligence.",
    ]

    scores = classify_yes_no(queries, documents, instruction=task)

    console.print("\n[bold green]Yes/No Classification Results[/bold green]")
    table = Table(show_header=True, header_style="bold magenta", show_lines=True)
    table.add_column("#", justify="center", style="dim", width=3)
    table.add_column("Query", style="cyan", max_width=35)
    table.add_column("Document", style="white", max_width=55)
    table.add_column("Label", justify="center", width=8)
    table.add_column("Score", justify="right", width=8)

    for idx, (query, doc, score) in enumerate(zip(queries, documents, scores), start=1):
        label = "YES" if score > 0.5 else "NO"
        l_style = "bold green" if score > 0.5 else "dim red"

        table.add_row(
            str(idx),
            query,
            doc[:80] + ("..." if len(doc) > 80 else ""),
            f"[{l_style}]{label}[/{l_style}]",
            f"[{l_style}]{score:.1f}[/{l_style}]",
        )

    console.print(table)
