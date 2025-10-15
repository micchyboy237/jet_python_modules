from typing import Dict
import spacy
from transformers import pipeline

nlp = spacy.load("en_core_web_sm")  # Small model; use 'lg' for better accuracy
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")  # Swap for custom fine-tuned model

def classify_sentence(sentence: str) -> Dict[str, str]:
    """Classify a sentence across categories. Returns dict with keys from summary."""
    doc = nlp(sentence)
    
    # By function (rule-based)
    func = "Declarative"
    if sentence.strip().endswith("?"):
        func = "Interrogative"
    elif sentence.strip().endswith("!") or doc[0].dep_ == "ROOT" and doc[0].pos_ == "VERB":  # Imperative often verb-initial
        func = "Imperative" if not sentence.strip().endswith("!") else "Exclamatory"
    
    # By structure (count clauses via sdep/conj)
    clauses = sum(1 for token in doc if token.dep_ in ("ccomp", "acl", "relcl")) + 1  # +1 for main
    struct = "Simple" if clauses == 1 else "Compound" if any(t.dep_ == "conj" for t in doc) else "Complex"
    
    # By components (voice)
    has_nsubj = any(token.dep_ == "nsubj" for token in doc)
    has_nsubjpass = any(token.dep_ == "nsubjpass" for token in doc)
    has_auxpass = any(token.dep_ == "auxpass" for token in doc)
    comp = "Passive" if has_nsubjpass or has_auxpass else "Active" if has_nsubj else "Unknown"
    
    # By connectors (example: contrastive)
    conn = "Contrastive" if any(token.text.lower() in {"but", "however"} for token in doc) else "None"
    
    # By meaning/brevity (ML embedding + simple rules)
    brevity = "One-word" if len(doc) == 1 else "Minor" if clauses < 1 else "Full"
    meaning_score = classifier(sentence)[0]  # Placeholder; fine-tune for causal/etc.
    meaning = "Affirmative" if meaning_score["label"] == "POSITIVE" else "Negative"  # Extend with custom labels
    
    return {
        "function": func, "structure": struct, "brevity": brevity,
        "meaning": meaning, "components": comp, "connectors": conn
    }

# Example usage
if __name__ == "__main__":
    sentences = ["She runs quickly, but he walks slowly.", "What a day!", "Close the door.", "If it rains, we stay home."]
    for s in sentences:
        print(f"{s}: {classify_sentence(s)}")