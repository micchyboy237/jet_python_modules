import spacy

# Initialize a blank English pipeline
nlp = spacy.blank("en")

# Add GLiNER component with no predefined labels
nlp.add_pipe("gliner_spacy", config={
    "model": "urchade/gliner_large-v2.1",
    "labels": ["entity"],   # or even []
    "threshold": 0.5
})

doc = nlp("""
The Barbie movie directed by Greta Gerwig became a cultural phenomenon in 2023.
Microsoft invested billions into OpenAI around the same time.
""")

for ent in doc.ents:
    print(ent.text, ent.label_)
