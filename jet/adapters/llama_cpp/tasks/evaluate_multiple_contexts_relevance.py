"""Evaluate relevance of multiple contexts using embedding-based logistic regression.

Equivalent to jet/llm/mlx/tasks/eval/evaluate_multiple_contexts_relevance.py.
Uses embed_utils for pair embeddings and sklearn LogisticRegression for classification.
The classifier is trained locally on CPU — no GPU/server required for classification.
"""

import os
from typing import Literal, Optional, TypedDict

import joblib
import numpy as np
from jet.adapters.llama_cpp.config import EMBED_MODEL
from jet.adapters.llama_cpp.embed_utils import embed
from jet.logger import logger
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


class ContextRelevanceResult(TypedDict):
    context: str
    relevance_score: Literal[0, 1, 2]
    score: float
    probabilities: list[float]
    is_valid: bool
    error: Optional[str]
    priority: Literal["low", "medium", "high"]


PRIORITY_MAP = {0: "low", 1: "medium", 2: "high"}

DEFAULT_PAIRS = [
    "Query: What is the capital of France?\nContext: The capital of France is Paris.",
    "Query: What is the capital of France?\nContext: Paris is a popular tourist destination.",
    "Query: What is the capital of France?\nContext: Einstein developed the theory of relativity.",
]
DEFAULT_LABELS = [2, 1, 0]


def _build_pairs(query: str, contexts: list[str]) -> list[str]:
    return [f"Query: {query}\nContext: {c}" for c in contexts]


def train_classifier(
    example_pairs: Optional[list[str]] = None,
    labels: Optional[list[int]] = None,
    model: str | None = None,
    verbose: bool = True,
) -> tuple[LogisticRegression, LabelEncoder]:
    """Train a logistic regression classifier on embedded query-context pairs."""
    resolved_model = model or EMBED_MODEL
    pairs = example_pairs or DEFAULT_PAIRS
    lbls = labels or DEFAULT_LABELS

    if len(pairs) != len(lbls):
        raise ValueError("Number of example pairs must match number of labels")

    logger.info(f"Training classifier: {len(pairs)} pairs, model={resolved_model}")

    embeddings = embed(
        pairs, model=resolved_model, return_format="numpy", show_progress=verbose
    )
    le = LabelEncoder()
    encoded = le.fit_transform(lbls)

    clf = LogisticRegression(solver="lbfgs", max_iter=200)
    clf.fit(embeddings, encoded)

    logger.info("Classifier training complete")
    return clf, le


def load_or_train_classifier(
    save_dir: Optional[str] = None,
    example_pairs: Optional[list[str]] = None,
    labels: Optional[list[int]] = None,
    model: str | None = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> tuple[LogisticRegression, LabelEncoder]:
    """Load cached classifier or train a new one."""
    if save_dir and not overwrite:
        clf_path = os.path.join(save_dir, "classifier.joblib")
        le_path = os.path.join(save_dir, "label_encoder.joblib")
        if os.path.isfile(clf_path) and os.path.isfile(le_path):
            logger.info(f"Loading classifier from {save_dir}")
            return joblib.load(clf_path), joblib.load(le_path)

    clf, le = train_classifier(example_pairs, labels, model, verbose)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(clf, os.path.join(save_dir, "classifier.joblib"))
        joblib.dump(le, os.path.join(save_dir, "label_encoder.joblib"))
        logger.info(f"Classifier saved to {save_dir}")

    return clf, le


def evaluate_multiple_contexts_relevance(
    query: str,
    contexts: list[str],
    model: str | None = None,
    save_dir: Optional[str] = None,
    example_pairs: Optional[list[str]] = None,
    labels: Optional[list[int]] = None,
    verbose: bool = True,
) -> list[ContextRelevanceResult]:
    """Evaluate relevance of multiple contexts for a query using embedding-based classification.

    Args:
        query: Search query to evaluate against.
        contexts: List of context strings to score.
        model: Embedding model key. Defaults to EMBED_MODEL.
        save_dir: Directory to cache/load classifier artifacts.
        example_pairs: Training pairs for classifier (uses defaults if None).
        labels: Labels for training pairs (uses defaults if None).
        verbose: Enable progress bars and detailed logging.

    Returns:
        List of ContextRelevanceResult sorted by (relevance_score, score) descending.
    """
    resolved_model = model or EMBED_MODEL

    if not query.strip():
        raise ValueError("Query cannot be empty.")
    if not contexts:
        raise ValueError("Contexts list cannot be empty.")

    logger.info(
        f"evaluate_multiple_contexts_relevance: {len(contexts)} contexts, "
        f"model={resolved_model}, query='{query[:60]}...'"
    )

    clf, le = load_or_train_classifier(
        save_dir=save_dir,
        example_pairs=example_pairs,
        labels=labels,
        model=resolved_model,
        verbose=verbose,
    )

    pairs = _build_pairs(query, contexts)
    embeddings = embed(
        pairs, model=resolved_model, return_format="numpy", show_progress=verbose
    )

    pred_probas = clf.predict_proba(embeddings)
    pred_indices = np.argmax(pred_probas, axis=1)
    scores = pred_probas[np.arange(len(pred_indices)), pred_indices]

    results: list[ContextRelevanceResult] = []
    for i, context in enumerate(contexts):
        try:
            predicted_label = int(le.inverse_transform([pred_indices[i]])[0])
            is_valid = predicted_label in (0, 1, 2)
            if not is_valid:
                logger.warning(
                    f"Invalid predicted label {predicted_label}, defaulting to 0"
                )
                predicted_label = 0

            results.append(
                ContextRelevanceResult(
                    context=context,
                    relevance_score=predicted_label,
                    score=float(scores[i]),
                    probabilities=pred_probas[i].tolist(),
                    is_valid=is_valid,
                    error=None if is_valid else f"Invalid label: {predicted_label}",
                    priority=PRIORITY_MAP[predicted_label],
                )
            )
        except Exception as e:
            logger.error(f"Error processing context '{context[:80]}': {e}")
            results.append(
                ContextRelevanceResult(
                    context=context,
                    relevance_score=0,
                    score=0.0,
                    probabilities=[0.0, 0.0, 0.0],
                    is_valid=False,
                    error=str(e),
                    priority="low",
                )
            )

    results.sort(key=lambda x: (x["relevance_score"], x["score"]), reverse=True)
    logger.info(f"Evaluation complete: {len(results)} results returned")
    return results


if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table

    console = Console()

    query = "What is the capital of France?"
    contexts = [
        "The capital of France is Paris.",
        "Paris is a popular tourist destination.",
        "Einstein developed the theory of relativity.",
    ]

    results = evaluate_multiple_contexts_relevance(query, contexts, verbose=True)

    console.print("\n[bold green]Multiple Contexts Relevance Results[/bold green]")
    table = Table(show_header=True, header_style="bold magenta", show_lines=True)
    table.add_column("#", justify="center", style="dim", width=3)
    table.add_column("Context", style="white", no_wrap=False, max_width=55)
    table.add_column("Score", justify="center", width=8)
    table.add_column("Confidence", justify="right", width=10)
    table.add_column("P(0)", justify="right", style="dim red", width=7)
    table.add_column("P(1)", justify="right", style="yellow", width=7)
    table.add_column("P(2)", justify="right", style="bold green", width=7)
    table.add_column("Priority", justify="center", width=9)

    score_styles = {0: "dim red", 1: "yellow", 2: "bold green"}
    priority_styles = {"low": "dim red", "medium": "yellow", "high": "bold green"}

    for rank, r in enumerate(results, start=1):
        s_style = score_styles.get(r["relevance_score"], "dim")
        p_style = priority_styles.get(r["priority"], "dim")
        probs = r["probabilities"]

        table.add_row(
            str(rank),
            r["context"][:80] + ("..." if len(r["context"]) > 80 else ""),
            f"[{s_style}]{r['relevance_score']}[/{s_style}]",
            f"{r['score']:.4f}",
            f"{probs[0]:.3f}" if len(probs) > 0 else "-",
            f"{probs[1]:.3f}" if len(probs) > 1 else "-",
            f"{probs[2]:.3f}" if len(probs) > 2 else "-",
            f"[{p_style}]{r['priority']}[/{p_style}]",
        )

    console.print(table)
