import spacy
from jet.logger import logger
from jet.transformers.formatters import format_json

# Initialize a blank English pipeline
nlp = spacy.blank("en")

# Add GLiNER component with correct config
nlp.add_pipe("gliner_spacy", config={
    "gliner_model": "urchade/gliner_large-v2.1",  # Correct key
    "labels": ["entity"],   # or even []
    "threshold": 0.1,
    "style": "ent"
})

text = """
Apple announced the new iPhone 16 during its annual event in Cupertino.
Elon Musk responded on X, saying Tesla's new battery tech will outperform everyone.
"""

doc = nlp(text)

entities = [{
    "text": entity.text,
    "label": entity.label_,
    "score": f"{entity._.score:.4f}"
} for entity in doc.ents]

logger.gray(f"RESULT ({len(entities)}):")
logger.success(format_json(entities))
