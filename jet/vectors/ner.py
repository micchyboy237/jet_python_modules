import hashlib
import json
from typing import Literal, TypedDict

import spacy
import torch
from shared.data_types.job import JobEntity

from jet.logger import logger

# Global cache for storing the loaded pipeline and its hash
nlp_cache = None
nlp_cache_hash = None

DEFAULT_GLNER_MODEL = "urchade/gliner_large-v2.1"
# Good default for this model (supports up to 8192 tokens)
# Balances speed, memory & entity-boundary quality on GTX 1660 / 16 GB RAM
DEFAULT_CHUNK_SIZE = 512

DeviceType = Literal["mps", "cpu", "auto"]


class Entity(TypedDict):
    text: str
    label: str
    score: float


def compute_config_hash(config: dict) -> str:
    """Compute a hash for the given configuration dictionary."""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()


def load_nlp_pipeline(
    labels: list[str],
    style: str = "ent",
    model: str = DEFAULT_GLNER_MODEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    device: DeviceType = "auto",
) -> spacy.language.Language:
    global nlp_cache, nlp_cache_hash
    custom_spacy_config = {
        "gliner_model": model,
        "chunk_size": chunk_size,
        "labels": labels,
        "style": style,
    }

    if device == "auto":
        if torch.backends.mps.is_available():
            map_loc = "mps"
        else:
            map_loc = "cpu"
    else:
        map_loc = device
    custom_spacy_config["map_location"] = map_loc

    new_hash = compute_config_hash(custom_spacy_config)

    if nlp_cache is None or nlp_cache_hash != new_hash:
        if not nlp_cache:
            logger.orange("Creating nlp_cache...")
        else:
            logger.warning("Config changed, recreating nlp_cache...")
        nlp_cache = spacy.blank("en")
        nlp_cache.add_pipe("gliner_spacy", config=custom_spacy_config)
        nlp_cache_hash = new_hash
    else:
        logger.debug("Reusing nlp_cache")

    return nlp_cache


def merge_dot_prefixed_words(text: str) -> str:
    tokens = text.split()
    merged_tokens = []
    for i, token in enumerate(tokens):
        if (
            token.startswith(".")
            and merged_tokens
            and not merged_tokens[-1].startswith(".")
        ):
            merged_tokens[-1] += token
        elif merged_tokens and merged_tokens[-1].endswith("."):
            merged_tokens[-1] += token
        else:
            merged_tokens.append(token)
    return " ".join(merged_tokens)


def extract_entities(nlp, text: str, threshold: float = 0.0) -> list[Entity]:
    with torch.inference_mode():
        doc = nlp(text)
        entities = [
            {
                "text": merge_dot_prefixed_words(entity.text),
                "label": entity.label_,
                "score": float(entity._.score),
            }
            for entity in doc.ents
            if float(entity._.score) >= threshold
        ]
        return entities


def extract_entities_from_text(nlp, text: str, threshold: float = 0.0) -> JobEntity:
    results = extract_entities(nlp, text, threshold=threshold)

    entities_dict = {}
    for entity in results:
        label = entity["label"].lower().replace(" ", "_")
        if label not in entities_dict:
            entities_dict[label] = []
        if entity["text"] not in entities_dict[label]:
            entities_dict[label].append(entity["text"])

    return entities_dict
