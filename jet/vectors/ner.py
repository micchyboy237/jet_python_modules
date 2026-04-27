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

DEFAULT_GLINER_MODEL = "urchade/gliner_large-v2.1"
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
    model: str = DEFAULT_GLINER_MODEL,
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


if __name__ == "__main__":
    labels = [
        "job title",  # Official position name (e.g. "Senior Software Engineer", "Marketing Manager")
        "company name",  # Hiring organization (e.g. "Google", "Acme Corp")
        "job location",  # Work location or setup (e.g. "New York, NY", "Remote", "London, UK")
        "salary range",  # Compensation details (e.g. "$120,000 - $160,000", "€65k–€85k per year")
        "experience level",  # Seniority or years of experience (e.g. "Senior", "Entry-level", "3–5 years")
        "employment type",  # Nature of employment (e.g. "Full-time", "Part-time", "Contract", "Internship")
        "required skills",  # Core skills (technical or soft) needed for the role (e.g. "Python, SQL", "communication")
        "technology stack",  # Specific tools, frameworks, languages, or platforms (e.g. "React, Node.js, AWS, Docker")
        "key responsibilities",  # Main duties and tasks (usually action-based descriptions of what the role does)
        "job requirements",  # Mandatory qualifications (e.g. "Bachelor's degree", "5+ years experience", "PMP certified")
        "employee benefits",  # Perks and benefits (e.g. "health insurance", "401(k)", "paid time off")
        "how to apply",  # Instructions for applying (e.g. "Send resume to email", "Apply via website")
        "application link",  # Direct application URL (e.g. "https://company.com/jobs/apply/12345")
        "work schedule",  # Working hours or pattern (e.g. "9am–6pm", "Monday–Friday", "shifts")
    ]
    nlp = load_nlp_pipeline(labels)
    sample_text = (
        "We are looking for a Senior Python Developer at TechCorp located in New York, NY. "
        "This is a full-time position with a salary range of $120,000 - $140,000. "
        "Required skills include Python, Django, and AWS. Benefits include health insurance, 401(k), and paid time off."
    )
    entities = extract_entities_from_text(nlp, sample_text, threshold=0.3)
    print("Extracted entities:")
    print(entities)
