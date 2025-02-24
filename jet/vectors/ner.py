from typing import List, Dict, TypedDict
from shared.data_types.job import JobEntity
import spacy

# Global cache for storing the loaded pipeline
nlp_cache = None


class Entity(TypedDict):
    text: str
    label: str
    score: float


def load_nlp_pipeline(model: str, labels: List[str], style: str, chunk_size: int):
    global nlp_cache
    if nlp_cache is None:
        custom_spacy_config = {
            "gliner_model": model,
            "chunk_size": chunk_size,
            "labels": labels,
            "style": style
        }
        nlp_cache = spacy.blank("en")
        nlp_cache.add_pipe("gliner_spacy", config=custom_spacy_config)
    return nlp_cache


def merge_dot_prefixed_words(text: str) -> str:
    tokens = text.split()
    merged_tokens = []
    for i, token in enumerate(tokens):
        if token.startswith(".") and merged_tokens and not merged_tokens[-1].startswith("."):
            merged_tokens[-1] += token
        elif merged_tokens and merged_tokens[-1].endswith("."):
            merged_tokens[-1] += token
        else:
            merged_tokens.append(token)
    return " ".join(merged_tokens)


def get_unique_entities(entities: List[Entity]) -> List[Entity]:
    best_entities: Dict[str, Entity] = {}
    for entity in entities:
        text = entity["text"]
        words = [t.replace(" ", "") for t in text.split(" ") if t]
        normalized_text = " ".join(words)
        label = entity["label"]
        score = float(entity["score"])
        entity["text"] = normalized_text
        key = f"{label}-{str(normalized_text)}"
        if key not in best_entities or score > float(best_entities[key]["score"]):
            entity["score"] = score
            best_entities[key] = entity
    return list(best_entities.values())


def extract_entities_from_text(nlp, text: str) -> JobEntity:
    doc = nlp(text)
    results = get_unique_entities([
        {
            "text": merge_dot_prefixed_words(entity.text),
            "label": entity.label_,
            "score": float(entity._.score)
        } for entity in doc.ents
    ])

    entities_dict = {}

    # Loop through the results
    for entity in results:
        # Get the label for grouping (assuming it's the 'label' label or another key)
        label = entity['label']
        label = label.lower().replace(" ", "_")

        # If the label key does not exist in the dictionary, create it
        if label not in entities_dict:
            entities_dict[label] = []

        # Append the entity to the respective label group
        if entity['text'] not in entities_dict[label]:
            entities_dict[label].append(entity['text'])

    return entities_dict
