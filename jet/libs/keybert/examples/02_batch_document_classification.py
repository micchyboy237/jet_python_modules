import pandas as pd
from jet.adapters.keybert import KeyBERT
from jet_telemetry import chain, get_trace_url, initialize_telemetry

# Initialize telemetry
initialize_telemetry(service_name="batch-classification-demo")

kw_model = KeyBERT()

# Sample dataset
documents = [
    "My login password is not working and I can't access my account.",
    "The app crashes every time I try to upload a photo.",
    "How do I upgrade my subscription plan to premium?",
    "I was charged twice for the same order this month.",
    "The new feature for dark mode is really impressive.",
]


@chain(name="classify-support-tickets")
def process_tickets(docs):
    # Extract keywords for ALL documents at once (optimized by jet adapter)
    all_keywords = kw_model.extract_keywords(
        docs, keyphrase_ngram_range=(1, 2), top_n=3, stop_words="english"
    )

    # Create a DataFrame for easy viewing
    df = pd.DataFrame(
        {
            "Document": docs,
            "Keywords": [", ".join([k[0] for k in kws]) for kws in all_keywords],
        }
    )
    return df


df = process_tickets(documents)
print(df)
print(f"\nTrace URL: {get_trace_url()}")
