import spacy
import torch
from jet.logger import logger

# Device selection
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# # Initialize blank pipeline and add GLiNER
# nlp = spacy.blank("en")
# nlp.add_pipe("gliner_spacy", config={
#     "gliner_model": "urchade/gliner_large-v2.1",
#     "labels": ["entity"],
#     "threshold": 0.1,
#     "style": "ent",
#     "map_location": device,
# })

# texts = [
#     "Elon Musk unveiled Tesla’s Cybertruck in Los Angeles in 2019.",
#     "Apple launched the Vision Pro at WWDC 2023 in Cupertino."
# ]

# # Process texts in bulk using nlp.pipe() for efficiency
# docs = list(nlp.pipe(texts))

# # Extract entities from all docs
# entities = []
# for doc in docs:
#     for entity in doc.ents:
#         entities.append({
#             "text": entity.text,
#             "label": entity.label_,
#             "score": f"{entity._.score:.4f}" if hasattr(entity._, 'score') else None
#         })

# logger.gray(f"RESULT ({len(entities)}):")
# logger.success(format_json(entities))


nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("gliner_spacy")

text = "This is a text about Bill Gates and Microsoft."
doc = nlp(text)

logger.debug(f"RESULTS ({len(doc.ents)}):")
for ent in doc.ents:
    logger.success(ent.text, ' => ', ent.label_, ' => ', ent._.score)

from spacy import displacy
import webbrowser

webbrowser.open("http://0.0.0.0:5001")
displacy.serve(doc, style="ent", port=5001)
