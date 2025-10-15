from typing import Dict, List, Literal, Tuple
import spacy
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load spaCy model (small for speed; use 'en_core_web_lg' for better accuracy)
nlp = spacy.load("en_core_web_sm")

def extract_features(sentence: str) -> np.ndarray:
    """Extract features for LogisticRegression from a sentence."""
    doc = nlp(sentence)
    
    # Features: sentence length, question mark, verb-initial, exclamation mark
    features = [
        len(doc),  # Sentence length (for brevity/function)
        1 if sentence.strip().endswith("?") else 0,  # Interrogative
        1 if doc[0].pos_ == "VERB" and doc[0].dep_ == "ROOT" else 0,  # Imperative
        1 if sentence.strip().endswith("!") else 0,  # Exclamatory
        sum(1 for token in doc if token.dep_ in ("ccomp", "acl", "relcl"))  # Clause count
    ]
    return np.array(features)

def train_classifier(
    sentences: List[str], labels: List[Literal["Declarative", "Interrogative", "Imperative", "Exclamatory"]]
) -> Tuple[LogisticRegression, StandardScaler]:
    """Train a LogisticRegression model for sentence function classification."""
    # Extract features
    X = np.array([extract_features(s) for s in sentences])
    
    # Scale features (important for LogisticRegression)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = LogisticRegression(multi_class="ovr", random_state=42, max_iter=200)
    model.fit(X_scaled, labels)
    
    return model, scaler

def classify_sentence(
    sentence: str, model: LogisticRegression, scaler: StandardScaler
) -> Dict[str, str]:
    """Classify a sentence's function using the trained model."""
    X = extract_features(sentence).reshape(1, -1)
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    
    return {"function": prediction}

# Example usage
if __name__ == "__main__":
    # Training data
    training_sentences = [
        "She runs quickly.",  # Declarative
        "What is your name?",  # Interrogative
        "Close the door!",  # Imperative
        "What a day!",  # Exclamatory
        "If it rains, we stay home.",  # Declarative
        "Is it sunny outside?",  # Interrogative
    ]
    training_labels = [
        "Declarative", "Interrogative", "Imperative", "Exclamatory",
        "Declarative", "Interrogative"
    ]
    
    # Train model
    model, scaler = train_classifier(training_sentences, training_labels)
    
    # Test classification
    test_sentences = [
        "Go to the store.",
        "The sun sets slowly.",
        "How are you?",
        "Amazing!"
    ]
    for sentence in test_sentences:
        result = classify_sentence(sentence, model, scaler)
        print(f"{sentence}: {result}")