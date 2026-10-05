from jet.adapters.keybert import KeyBERT

# Initialize KeyBERT using the jet adapter (defaults to llama.cpp embed model)
kw_model = KeyBERT()

doc = """
    Artificial Intelligence is transforming the healthcare industry by enabling 
    predictive analytics, personalized treatment plans, and automated diagnostic tools. 
    Machine learning algorithms analyze vast amounts of patient data to identify patterns 
    that human doctors might miss.
"""

# 1. Extract Single Keywords (Unigrams)
print("--- Single Keywords ---")
keywords = kw_model.extract_keywords(
    doc, keyphrase_ngram_range=(1, 1), stop_words="english"
)
for keyword, score in keywords[:5]:
    print(f"{keyword}: {score:.4f}")

# 2. Extract Keyphrases (Bigrams/Trigrams)
print("\n--- Keyphrases ---")
keyphrases = kw_model.extract_keywords(
    doc, keyphrase_ngram_range=(1, 3), stop_words="english"
)
for phrase, score in keyphrases[:5]:
    print(f"{phrase}: {score:.4f}")
