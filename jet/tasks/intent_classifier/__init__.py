from typing import Dict, List, TypedDict, Union
from jet.logger import logger
from jet.models.model_registry.transformers.bert_model_registry import BERTModelRegistry, ONNXBERTWrapper
from transformers import PreTrainedTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from tqdm import tqdm
from .utils import ClassificationResult, Id2Label, transform_label


def classify_intents(
    model: Union[AutoModelForSequenceClassification, ONNXBERTWrapper],
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
    batch_size: int = 32,
    show_progress: bool = False,
) -> List[ClassificationResult]:
    """
    Classify intents for a list of texts using a BERT model.

    Args:
        model: The BERT model (PyTorch or ONNX).
        tokenizer: The tokenizer for the model.
        texts: List of texts to classify.
        batch_size: Size of batches for processing.
        show_progress: Whether to show a progress bar.

    Returns:
        List of classification results with label, score, value, text, doc_index, and rank.
    """
    logger.info(f"Classifying {len(texts)} texts with batch size {batch_size}")

    # Get id2label from model config or reconstruct from label2id
    id2label: Id2Label = (
        model.config.id2label
        if hasattr(model, "config") and model.config.id2label is not None
        else {str(v): k for k, v in model.config.label2id.items()}
        if hasattr(model, "config") and model.config.label2id is not None
        else {}
    )

    if not id2label:
        logger.warning(
            "No id2label or label2id found in model config. Using default LABEL_{id} format.")

    results = []
    if isinstance(model, ONNXBERTWrapper):
        logger.debug("Using ONNXBERTWrapper for classification")
        logits = model.classify(texts, batch_size=batch_size)
        scores = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        iterator = tqdm(range(len(scores)),
                        desc="Processing texts", disable=not show_progress)
        for i in iterator:
            label_id = np.argmax(scores[i])
            label = transform_label(
                label_id, id2label) if id2label else f"LABEL_{label_id}"
            result = {
                "doc_index": i,
                "rank": 0,
                "label": label,
                "value": int(label_id),
                "score": float(scores[i][label_id]),
                "text": texts[i],
            }
            logger.debug(
                f"Text {i}: {texts[i]}, Label: {label}, Score: {float(scores[i][label_id]):.4f}, Value: {label_id}"
            )
            results.append(result)
    else:
        logger.debug(
            "Using AutoModelForSequenceClassification for classification")
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches", disable=not show_progress):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(model.device)
            with torch.no_grad():
                outputs = model(**inputs)
            scores = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            for j, score in enumerate(scores):
                label_id = np.argmax(score)
                label = transform_label(
                    label_id, id2label) if id2label else f"LABEL_{label_id}"
                result = {
                    "doc_index": i + j,
                    "rank": 0,
                    "label": label,
                    "value": int(label_id),
                    "score": float(score[label_id]),
                    "text": batch_texts[j],
                }
                logger.debug(
                    f"Text {i + j}: {batch_texts[j]}, Label: {label}, Score: {float(score[label_id]):.4f}, Value: {label_id}"
                )
                results.append(result)

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    for idx, result in enumerate(results):
        result["rank"] = idx + 1
    logger.info(f"Classification completed, processed {len(results)} results")
    return results
