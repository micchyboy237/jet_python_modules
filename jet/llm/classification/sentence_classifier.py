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
    elif sentence.strip().endswith("!") or (doc[0].dep_ == "ROOT" and doc[0].pos_ == "VERB"):  # Imperative often verb-initial
        func = "Imperative" if not sentence.strip().endswith("!") else "Exclamatory"
    
    # By structure (count clauses via ccomp/acl/relcl/advcl and check conj)
    clauses = sum(1 for token in doc if token.dep_ in ("ccomp", "acl", "relcl", "advcl")) + 1  # +1 for main
    has_conj = any(t.dep_ == "conj" or t.text.lower() in {"and", "but", "or"} for t in doc)
    struct = "Compound" if has_conj else "Simple" if clauses == 1 else "Complex"
    
    # By components (voice)
    comp = "Active"  # Default to Active
    if any(token.dep_ == "nsubjpass" for token in doc):  # Passive if nsubjpass exists
        comp = "Passive"
    elif any(token.dep_ == "nsubj" and token not in [t for t in doc if t.dep_ == "pobj"] for token in doc):
        comp = "Active"
    
    # By connectors (example: contrastive)
    conn = "Contrastive" if any(token.text.lower() in {"but", "however"} for token in doc) else "None"
    
    # By meaning/brevity (ML embedding + simple rules)
    brevity = "One-word" if len(sentence.strip().split()) == 1 else "Minor" if clauses < 1 else "Full"
    meaning_score = classifier(sentence)[0]  # Placeholder; fine-tune for causal/etc.
    meaning = "Affirmative" if meaning_score["label"] == "POSITIVE" else "Negative"  # Extend with custom labels
    
    return {
        "function": func, "structure": struct, "brevity": brevity,
        "meaning": meaning, "components": comp, "connectors": conn
    }
