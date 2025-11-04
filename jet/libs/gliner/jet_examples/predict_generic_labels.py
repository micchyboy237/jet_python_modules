from gliner import GLiNER

# Load a pretrained GLiNER model (CPU-friendly)
model = GLiNER.from_pretrained("urchade/gliner_large-v2.1")

text = """
Apple announced the new iPhone 16 during its annual event in Cupertino.
Elon Musk responded on X, saying Tesla's new battery tech will outperform everyone.
"""

# When you don't know the labels — use a generic placeholder
generic_labels = ["entity"]  # open ontology mode

# Predict entities
entities = model.predict_entities(text, generic_labels, threshold=0.5)

for ent in entities:
    print(ent)
