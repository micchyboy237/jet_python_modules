import spacy
import torch
from jet.logger import logger
from jet.transformers.formatters import format_json

# Device selection
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

nlp = spacy.load("en_core_web_md")
nlp.add_pipe("gliner_spacy")

texts = [
    "Elon Musk unveiled Tesla’s Cybertruck in Los Angeles in 2019.",
    "Apple launched the Vision Pro at WWDC 2023 in Cupertino."
]

# Process texts in bulk using nlp.pipe() for efficiency
docs = list(nlp.pipe(texts))

# Extract entities from all docs
entities = []
for doc in docs:
    for entity in doc.ents:
        entities.append({
            "text": entity.text,
            "label": entity.label_,
            "score": f"{entity._.score:.4f}" if hasattr(entity._, 'score') else None
        })

logger.gray(f"RESULT ({len(entities)}):")
logger.success(format_json(entities))

from spacy import displacy
import webbrowser

webbrowser.open("http://0.0.0.0:5001")
displacy.serve(docs, style="ent", port=5001)
