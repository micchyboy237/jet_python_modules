"""Classify question-context pairs into labels using embedding-based logistic regression.

Equivalent to jet/llm/mlx/tasks/answer_multiple_labels_with_context.py.
Uses embed_utils.embed() for pair embeddings and sklearn LogisticRegression
for classification. The LLM model from the MLX version is unused for generation;
classification is purely embedding-based, so we skip LLM loading entirely.
"""

import os
from typing import Optional, TypedDict

import joblib
import numpy as np
from jet.adapters.llama_cpp.config import EMBED_MODEL
from jet.adapters.llama_cpp.embed_utils import embed
from jet.logger import logger
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


class QuestionContext(TypedDict):
    question: str
    context: str


class AnswerResult(TypedDict):
    question: str
    context: str
    answer: str
    confidence: float
    is_valid: bool
    error: Optional[str]


DEFAULT_TRAIN_PAIRS = [
    "Context: The movie received widespread acclaim for its compelling story. Question: What is the sentiment of the movie review?",
    "Context: The product was described as unreliable and poorly designed. Question: What is the sentiment of the product review?",
    "Context: The book was neither particularly exciting nor boring. Question: What is the sentiment of the book review?",
]
DEFAULT_TRAIN_LABELS = ["Positive", "Negative", "Neutral"]


def _validate_inputs(
    questions_contexts: list[QuestionContext], labels: list[str]
) -> None:
    if not questions_contexts:
        raise ValueError("Questions and contexts list cannot be empty.")
    if not labels:
        raise ValueError("Labels list cannot be empty.")
    for qc in questions_contexts:
        if not qc.get("question", "").strip():
            raise ValueError(f"Question cannot be empty: {qc}")
        if not qc.get("context", "").strip():
            raise ValueError(f"Context cannot be empty for question: {qc['question']}")


def train_classifier(
    embedder_model: str | None = None,
    example_pairs: Optional[list[str]] = None,
    labels: Optional[list[str]] = None,
    verbose: bool = True,
) -> tuple[LogisticRegression, LabelEncoder]:
    """Train a logistic regression classifier on embedded question-context pairs."""
    resolved_model = embedder_model or EMBED_MODEL
    pairs = example_pairs or DEFAULT_TRAIN_PAIRS
    lbls = labels or DEFAULT_TRAIN_LABELS

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
    embedder_model: str | None = None,
    example_pairs: Optional[list[str]] = None,
    labels: Optional[list[str]] = None,
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

    clf, le = train_classifier(embedder_model, example_pairs, labels, verbose)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(clf, os.path.join(save_dir, "classifier.joblib"))
        joblib.dump(le, os.path.join(save_dir, "label_encoder.joblib"))
        logger.info(f"Classifier saved to {save_dir}")

    return clf, le


def answer_multiple_labels_with_context(
    questions_contexts: list[QuestionContext],
    labels: list[str],
    model: str | None = None,
    save_dir: Optional[str] = None,
    example_pairs: Optional[list[str]] = None,
    training_labels: Optional[list[str]] = None,
    batch_size: int = 32,
    verbose: bool = True,
) -> list[AnswerResult]:
    """Classify question-context pairs into labels using embedding-based classification.

    Args:
        questions_contexts: List of dicts with 'question' and 'context' keys.
        labels: Valid label strings (e.g., ['Positive', 'Negative', 'Neutral']).
        model: Embedding model key. Defaults to EMBED_MODEL.
        save_dir: Directory to cache/load classifier artifacts.
        example_pairs: Training pairs for classifier (uses defaults if None).
        training_labels: Labels for training pairs (uses defaults if None).
        batch_size: Batch size for embedding generation.
        verbose: Enable progress bars and detailed logging.

    Returns:
        List of AnswerResult dicts with predicted label, confidence, and validity.
    """
    resolved_model = model or EMBED_MODEL

    try:
        _validate_inputs(questions_contexts, labels)
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        raise

    logger.info(
        f"answer_multiple_labels: {len(questions_contexts)} items, "
        f"{len(labels)} labels, model={resolved_model}"
    )

    clf, le = load_or_train_classifier(
        save_dir=save_dir,
        embedder_model=resolved_model,
        example_pairs=example_pairs,
        labels=training_labels,
        verbose=verbose,
    )

    pairs = [
        f"Context: {qc['context']} Question: {qc['question']}"
        for qc in questions_contexts
    ]

    logger.debug(f"Embedding {len(pairs)} question-context pairs...")
    embeddings = embed(
        pairs,
        model=resolved_model,
        return_format="numpy",
        batch_size=batch_size,
        show_progress=verbose,
    )

    pred_probas = clf.predict_proba(embeddings)
    pred_indices = np.argmax(pred_probas, axis=1)
    confidences = pred_probas[np.arange(len(pred_indices)), pred_indices]

    results: list[AnswerResult] = []
    for i, qc in enumerate(questions_contexts):
        try:
            predicted_label = str(le.inverse_transform([pred_indices[i]])[0])
            confidence = float(confidences[i])
            is_valid = predicted_label in labels

            if not is_valid:
                logger.warning(
                    f"Invalid label '{predicted_label}' for question: {qc['question'][:60]}"
                )
                predicted_label = labels[0]

            results.append(
                AnswerResult(
                    question=qc["question"],
                    context=qc["context"],
                    answer=predicted_label,
                    confidence=confidence,
                    is_valid=is_valid,
                    error=None
                    if is_valid
                    else f"Predicted '{predicted_label}' not in {labels}",
                )
            )
        except Exception as e:
            logger.error(f"Error processing question '{qc['question'][:60]}': {e}")
            results.append(
                AnswerResult(
                    question=qc["question"],
                    context=qc["context"],
                    answer=labels[0] if labels else "",
                    confidence=0.0,
                    is_valid=False,
                    error=str(e),
                )
            )

    valid_count = sum(1 for r in results if r["is_valid"])
    logger.info(
        f"Classification complete: {valid_count}/{len(results)} valid predictions"
    )
    return results


if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table

    console = Console()

    questions_contexts = [
        {
            "question": "What is the sentiment of the movie review?",
            "context": "The movie was thrilling and well-acted.",
        },
        {
            "question": "What is the sentiment of the product review?",
            "context": "The product broke after one use.",
        },
        {
            "question": "What is the sentiment of the book review?",
            "context": "The book was average, with nothing memorable.",
        },
    ]
    labels = ["Positive", "Negative", "Neutral"]

    results = answer_multiple_labels_with_context(
        questions_contexts, labels, verbose=True
    )

    console.print("\n[bold green]Multiple Labels Classification Results[/bold green]")
    table = Table(show_header=True, header_style="bold magenta", show_lines=True)
    table.add_column("#", justify="center", style="dim", width=3)
    table.add_column("Question", style="cyan", max_width=35)
    table.add_column("Context", style="white", max_width=40)
    table.add_column("Label", justify="center", width=12)
    table.add_column("Confidence", justify="right", width=11)
    table.add_column("Valid", justify="center", width=6)

    label_styles = {
        "Positive": "bold green",
        "Negative": "bold red",
        "Neutral": "yellow",
    }

    for idx, r in enumerate(results, start=1):
        l_style = label_styles.get(r["answer"], "dim")
        v_style = "bold green" if r["is_valid"] else "bold red"
        v_str = "✓" if r["is_valid"] else "✗"

        err = f"\n[dim red]⚠ {r['error']}[/dim red]" if not r["is_valid"] else ""

        table.add_row(
            str(idx),
            r["question"],
            r["context"][:60] + ("..." if len(r["context"]) > 60 else ""),
            f"[{l_style}]{r['answer']}[/{l_style}]",
            f"{r['confidence']:.4f}",
            f"[{v_style}]{v_str}[/{v_style}]{err}",
        )

    console.print(table)
