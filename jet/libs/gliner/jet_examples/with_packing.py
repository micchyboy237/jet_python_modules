# with_packing.py
from gliner import GLiNER, InferencePackingConfig

model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
model.eval()

texts = [
    "Elon Musk unveiled Tesla’s Cybertruck in Los Angeles in 2019.",
    "Apple launched the Vision Pro at WWDC 2023 in Cupertino."
]

labels = ["Person", "Organization", "Location", "Event", "Date"]

# Enable packing: combine texts into longer sequences
packing_config = InferencePackingConfig(
    max_length=512,           # Adjust based on model.config.max_len
    streams_per_batch=8,      # More streams = better packing efficiency
    sep_token_id=model.data_processor.transformer_tokenizer.eos_token_id
)

predictions = model.run(
    texts,
    labels,
    batch_size=1,
    threshold=0.5,
    packing_config=packing_config
)

for text, ents in zip(texts, predictions):
    print(f"Text: {text}")
    for e in ents:
        print(f"  → '{e['text']}' | {e['label']} | {e['score']:.2f}")
    print()