from typing import Dict, TypedDict

# Remove IntentLabel and LABELS
Id2Label = Dict[int, str]  # Dynamic labels as strings


def transform_label(index: int, id2label: Id2Label) -> str:
    """
    Transform a label index to its corresponding label using the id2label mapping.

    Args:
        index: The integer index of the label.
        id2label: Dictionary mapping string indices to label names.

    Returns:
        The label string corresponding to the index.

    Raises:
        IndexError: If the index is not found in id2label.
    """
    try:
        return id2label[index]
    except KeyError:
        raise IndexError(
            f"Label index {index} is not found in id2label mapping")


class ClassificationResult(TypedDict):
    label: str  # Changed from IntentLabel to str
    score: float
    value: int
    text: str
    doc_index: int
    rank: int
