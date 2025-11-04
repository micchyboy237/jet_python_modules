from gliner import GLiNER

# Load model (runs on MPS if available, else CPU)
model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
model.eval()

texts = [
    "Elon Musk unveiled Tesla’s Cybertruck in Los Angeles in 2019.",
    "Apple launched the Vision Pro at WWDC 2023 in Cupertino."
]

labels = ["Person", "Organization", "Location", "Event", "Date"]

# Run without packing
predictions = model.run(texts, labels, batch_size=1, threshold=0.5, packing_config=None)

for text, ents in zip(texts, predictions):
    print(f"Text: {text}")
    for e in ents:
        print(f"  → '{e['text']}' | {e['label']} | {e['score']:.2f}")
    print()