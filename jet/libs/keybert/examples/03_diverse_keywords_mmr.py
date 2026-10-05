from jet.adapters.keybert import KeyBERT

kw_model = KeyBERT()

doc = """
    Python is a high-level, general-purpose programming language. 
    Its design philosophy emphasizes code readability with the use of significant indentation.
    Python is dynamically typed and garbage-collected. It supports multiple programming paradigms, 
    including structured, object-oriented and functional programming.
"""

# Standard Extraction
print("--- Standard Extraction ---")
std_kws = kw_model.extract_keywords(doc, top_n=5, stop_words="english")
print([k[0] for k in std_kws])

# MMR Extraction (Maximal Marginal Relevance)
print("\n--- MMR Extraction (Diverse) ---")
mmr_kws = kw_model.extract_keywords(
    doc, keyphrase_ngram_range=(1, 2), use_mmr=True, diversity=0.7, stop_words="english"
)
print([k[0] for k in mmr_kws])
