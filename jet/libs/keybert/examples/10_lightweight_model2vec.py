from jet.adapters.keybert import KeyBERT
from jet_telemetry import get_trace_url, initialize_telemetry

initialize_telemetry(service_name="lightweight-keybert")

# Initialize KeyBERT using the default jet adapter (nomic-embed via llama.cpp)
# This is already lightweight and fast compared to full BERT transformers
kw_model = KeyBERT()

doc = "Quantum computing will revolutionize cryptography."

keywords = kw_model.extract_keywords(doc)
print("--- Lightweight KeyBERT Keywords ---", keywords)

print(f"\nTrace URL: {get_trace_url()}")
