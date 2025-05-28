import os
from typing import Dict, List, Optional, TypedDict
import uuid
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sentence_transformers.util import cos_sim


class SentenceClassifier(nn.Module):
    def __init__(self, base_model: AutoModel, num_labels: int):
        super().__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(base_model.config.hidden_size, num_labels)

    def forward(self, inputs: Dict) -> torch.Tensor:
        outputs = self.base_model(**inputs).last_hidden_state
        cls_embedding = outputs[:, 0]  # CLS pooling
        logits = self.classifier(cls_embedding)
        return logits


# Load model and tokenizer
model_id = 'mixedbread-ai/mxbai-embed-large-v1'
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModel.from_pretrained(model_id)

# Check if MPS is available
device = torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device("cpu")
base_model = base_model.to(device)

# Load trained classifier
classifier = SentenceClassifier(base_model, num_labels=2).to(device)
classifier.load_state_dict(torch.load(
    f"{os.path.dirname(__file__)}/trained_sentence_classifier.pth"))
classifier.eval()

# Sample dataset for binary sentiment classification
classification_data = [
    ("The concert was absolutely phenomenal!", 1),  # Positive
    ("This book was a complete waste of time.", 0),  # Negative
    ("I had an incredible time at the party!", 1),   # Positive
    ("The service at this restaurant was awful.", 0),  # Negative
]

# Inference
print("\nClassification Results (CLS Pooling):")
with torch.no_grad():
    for text, true_label in classification_data:
        inputs = tokenizer(text, padding=True,
                           return_tensors='pt', truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = classifier(inputs)
        probs = torch.softmax(logits, dim=-1)
        predicted_label = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, predicted_label].item()
        print(f"Text: {text}")
        print(f"True Label: {'Positive' if true_label == 1 else 'Negative'}")
        print(
            f"Predicted Label: {'Positive' if predicted_label == 1 else 'Negative'}")
        print(f"Confidence: {confidence:.4f}")
        print()
