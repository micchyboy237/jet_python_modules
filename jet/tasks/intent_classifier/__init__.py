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
    logger.info(f"Classifying {len(texts)} texts with batch size {batch_size}")
    id2label: Id2Label = model.config.id2label if hasattr(
        model, "config") else None
    results = []
    if isinstance(model, ONNXBERTWrapper):
        logger.debug("Using ONNXBERTWrapper for classification")
        logits = model.classify(texts, batch_size=batch_size)
        scores = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        iterator = tqdm(range(len(scores)),
                        desc="Processing texts", disable=not show_progress)
        for i in iterator:
            label_id = np.argmax(scores[i])
            try:
                label = transform_label(
                    label_id) if id2label else f"LABEL_{label_id}"
            except IndexError:
                label = f"LABEL_{label_id}"
            result = {
                "doc_index": i,
                "rank": 0,  # Temporary rank, will be updated after sorting
                "label": label,
                "value": int(label_id),
                "score": float(scores[i][label_id]),
                "text": texts[i],
            }
            logger.debug(
                f"Text {i}: {texts[i]}, Label: {label}, Score: {float(scores[i][label_id]):.4f}, Value: {label_id}")
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
                try:
                    label = transform_label(
                        label_id) if id2label else f"LABEL_{label_id}"
                except IndexError:
                    label = f"LABEL_{label_id}"
                result = {
                    "doc_index": i + j,
                    "rank": 0,  # Temporary rank, will be updated after sorting
                    "label": label,
                    "value": int(label_id),
                    "score": float(score[label_id]),
                    "text": batch_texts[j],
                }
                logger.debug(
                    f"Text {i + j}: {batch_texts[j]}, Label: {label}, Score: {float(score[label_id]):.4f}, Value: {label_id}")
                results.append(result)
    # Sort results by score in descending order and assign ranks
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    for idx, result in enumerate(results):
        result["rank"] = idx + 1
    logger.info(f"Classification completed, processed {len(results)} results")
    return results
