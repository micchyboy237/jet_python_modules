from typing import List, Dict, Union, Optional
import coremltools as ct
import numpy as np
from jet.logger import logger
from jet.transformers.formatters import format_json


def classify_text_coreml(
    text: Union[str, List[str]],
    model_path: str = "~/.cache/huggingface/hub/models--Falconsai--intent_classification/snapshots/630d0d4668170a2a64d8d80b04d9844415bd4367/coreml/text-classification/float32_model.mlpackage",
    batch_size: Optional[int] = None
) -> List[Dict[str, Union[str, float]]]:
    """
    Classify text(s) using a Core ML model on Apple Silicon.

    Args:
        text: A single string or list of strings to classify.
        model_path: Path to the Core ML model package (.mlpackage).
        batch_size: Batch size for processing multiple texts (optional, used only for list input).

    Returns:
        List of dictionaries containing classification results (label and score).

    Raises:
        FileNotFoundError: If the model file does not exist.
        ValueError: If input text is invalid or model inference fails.
    """
    # Load Core ML model
    try:
        model = ct.models.MLModel(model_path)
        logger.gray(f"Loaded Core ML model from {model_path}")
    except FileNotFoundError:
        logger.error(f"Model file not found at {model_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to load Core ML model: {str(e)}")
        raise ValueError(f"Model loading failed: {str(e)}")

    # Prepare input
    texts = [text] if isinstance(text, str) else text
    if not texts:
        logger.error("Input text is empty")
        raise ValueError("Input text cannot be empty")

    # Process texts in batches
    results = []
    batch_size = batch_size or len(texts)
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            for single_text in batch:
                # Core ML models typically expect tokenized or preprocessed input
                # Assuming the model expects a string input (adjust based on model metadata)
                prediction = model.predict({"input_text": single_text})

                # Extract label and score (adjust keys based on model output)
                label = prediction["label"] if "label" in prediction else list(
                    prediction.values())[0]
                score = prediction.get("score", float(
                    prediction.get("probability", 1.0)))

                results.append({"label": str(label), "score": float(score)})
        except Exception as e:
            logger.error(
                f"Inference failed for batch {i//batch_size + 1}: {str(e)}")
            raise ValueError(f"Inference failed: {str(e)}")

    logger.gray("Classification result:")
    logger.success(format_json(results))
    return results
