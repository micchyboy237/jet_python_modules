import sys
import json
import spacy
from typing import List, Dict

nlp_cache = None


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


def determine_chunk_size(text: str) -> int:
    """Dynamically set chunk size based on text length."""
    length = len(text)
    if length < 1000:
        return 250
    elif length < 3000:
        return 350
    else:
        return 500


def main():
    model = sys.argv[1]
    texts = json.loads(sys.argv[2])  # List of texts
    labels = json.loads(sys.argv[3])
    style = sys.argv[4]

    for text in texts:
        chunk_size = determine_chunk_size(text)
        nlp = load_nlp_pipeline(model, labels, style, chunk_size)
        doc = nlp(text)
        entities = [{
            "text": entity.text,
            "label": entity.label_,
            "score": f"{entity._.score:.4f}"
        } for entity in doc.ents]

        print(f"result: {json.dumps({
            "text": text,
            "entities": entities
        })}")


if __name__ == "__main__":
    main()
