from jet.adapters.keybert import KeyBERT

kw_model = KeyBERT()

doc = """
    Climate change is causing rising sea levels and extreme weather events. 
    Scientists warn that immediate action is required to reduce carbon emissions.
"""

# Extract and highlight
highlighted_doc = kw_model.extract_keywords(doc, highlight=True)

print("--- Highlighted Document ---")
print(highlighted_doc)
