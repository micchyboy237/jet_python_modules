from gliner import GLiNER
import torch

from jet.logger import logger
from jet.transformers.formatters import format_json

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Load a pretrained GLiNER model (auto-selected device)
model = GLiNER.from_pretrained("urchade/gliner_large-v2.1", map_location=device)

texts = [
    "Elon Musk unveiled Tesla’s Cybertruck in Los Angeles in 2019.",
    "Apple launched the Vision Pro at WWDC 2023 in Cupertino."
]

# When you don't know the labels — use a generic placeholder
generic_labels = ["entity"]  # open ontology mode

# Predict entities
entities_lists = model.run(texts, generic_labels, threshold=0.1, multi_label=True)
entities = [ent for ents in entities_lists for ent in ents]

logger.gray(f"RESULT ({len(entities)}):")
logger.success(format_json(entities))
