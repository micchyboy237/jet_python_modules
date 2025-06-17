from typing import Dict, List, Literal, TypedDict

IntentLabel = Literal[
    "cancellation",
    "ordering",
    "shipping",
    "invoicing",
    "billing and payment",
    "returns and refunds",
    "complaints and feedback",
    "speak to person",
    "edit account",
    "delete account",
    "delivery information",
    "subscription",
    "recover password",
    "registration problems",
    "appointment"
]

LABELS: List[IntentLabel] = [
    "cancellation",
    "ordering",
    "shipping",
    "invoicing",
    "billing and payment",
    "returns and refunds",
    "complaints and feedback",
    "speak to person",
    "edit account",
    "delete account",
    "delivery information",
    "subscription",
    "recover password",
    "registration problems",
    "appointment"
]

Id2Label = Dict[str, IntentLabel]


def transform_label(index: int) -> IntentLabel:
    try:
        return LABELS[index]
    except IndexError:
        raise IndexError(
            f"Label index {index} is out of range for available labels")


class ClassificationResult(TypedDict):
    label: IntentLabel
    score: float
    value: int
