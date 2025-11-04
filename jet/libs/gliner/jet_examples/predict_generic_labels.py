from gliner import GLiNER
import torch

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
entities = model.run(texts, generic_labels, threshold=0.1, multi_label=True)

print(f"Entities: {len(entities)}")
for ent in entities:
    print(ent)
