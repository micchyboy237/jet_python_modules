from jet.adapters.keybert import KeyBERT

# A sample blog post snippet to test
blog_post = """
The shift toward renewable energy has accelerated rapidly over the past decade. 
With solar panels and wind turbines becoming cheaper to manufacture, cities worldwide 
are integrating sustainable power into their local grids. However, managing this 
intermittent energy supply requires advanced smart grid technology. By utilizing 
machine learning algorithms, these modernized power grids can predict peak energy demand, 
optimize battery storage distribution, and drastically reduce carbon emissions in real-time.
"""

# 1. Initialize KeyBERT
kw_model = KeyBERT()

print("--- Processing Blog Post ---")

# 2. Extract standard keywords (Single words)
# We filter out common English stop words (the, a, and, etc.)
basic_tags = kw_model.extract_keywords(
    blog_post, keyphrase_ngram_range=(1, 1), stop_words="english", top_n=5
)

print("\n💡 Standard Single-Word Tags:")
for tag, confidence in basic_tags:
    print(  # Just capitalizing for neat output
        f" - {tag.title()} (Confidence: {confidence:.4f})"
    )


# 3. Extract diverse keyphrases (1 to 2 word phrases)
# We use MMR here to make sure the tags cover different aspects of the text.
advanced_tags = kw_model.extract_keywords(
    blog_post,
    keyphrase_ngram_range=(1, 2),
    stop_words="english",
    use_mmr=True,  # Activates Maximal Marginal Relevance
    diversity=0.6,  # High values = more diverse tags; low values = more similar tags
    top_n=5,
)

print("\n🚀 Advanced Meta-Tags (Diversified & Multi-word):")
for tag, confidence in advanced_tags:
    print(f" - {tag.title()} (Confidence: {confidence:.4f})")
