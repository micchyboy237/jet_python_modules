from typing import Dict, List, Optional, TypedDict
import uuid
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sentence_transformers.util import cos_sim


class SimilarityResult(TypedDict):
    """
    Represents a single similarity result for a text.

    Fields:
        id: Identifier for the text.
        rank: Rank based on score (1 for highest score).
        doc_index: Index of the document in the input list.
        score: Similarity score between query and text.
        percent_difference: Percentage difference from the highest score, rounded to 2 decimals.
        text: The compared text.
    """
    id: str
    rank: int
    doc_index: int
    score: float
    percent_difference: Optional[float]
    text: str

# For retrieval you need to pass this prompt.


def transform_query(query: str) -> str:
    """ For retrieval, add the prompt for query (not for documents).
    """
    return f'Represent this sentence for searching relevant passages: {query}'

# Pooling function


def pooling(outputs: torch.Tensor, inputs: Dict, strategy: str = 'cls') -> np.ndarray:
    if strategy == 'cls':
        outputs = outputs[:, 0]
    elif strategy == 'mean':
        outputs = torch.sum(
            outputs * inputs["attention_mask"][:, :, None], dim=1) / torch.sum(inputs["attention_mask"], dim=1, keepdim=True)
    else:
        raise NotImplementedError
    return outputs.detach().cpu().numpy()

# Classifier model


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

# Compute similarity results for retrieval


def compute_similarity_results(embeddings: np.ndarray, docs: List[str], query_index: int = 0) -> List[SimilarityResult]:
    similarities = cos_sim(
        embeddings[query_index], embeddings[1:]).flatten().tolist()
    similarity_results: List[SimilarityResult] = []
    max_score = max(similarities) if similarities else 1.0

    for idx, (score, text) in enumerate(zip(similarities, docs[1:])):
        percent_diff = round(((max_score - score) / max_score)
                             * 100, 2) if max_score != 0 else 0.0
        similarity_results.append({
            "id": str(uuid.uuid4()),
            "rank": 0,
            "doc_index": idx + 1,
            "score": round(score, 4),
            "percent_difference": percent_diff,
            "text": text
        })

    similarity_results.sort(key=lambda x: x["score"], reverse=True)
    for rank, result in enumerate(similarity_results, 1):
        result["rank"] = rank
    return similarity_results


# Load model and tokenizer
model_id = 'mixedbread-ai/mxbai-embed-large-v1'
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModel.from_pretrained(model_id)

# Check if MPS is available
device = torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device("cpu")
base_model = base_model.to(device)

# --- Retrieval Task ---
docs = [
    transform_query('A man is eating a piece of bread'),
    "A man is eating food.",
    "A man is eating pasta.",
    "The girl is carrying a baby.",
    "A man is riding a horse.",
]

# Encode inputs for retrieval
inputs = tokenizer(docs, padding=True, return_tensors='pt')
inputs = {k: v.to(device) for k, v in inputs.items()}
outputs = base_model(**inputs).last_hidden_state
embeddings = pooling(outputs, inputs, 'cls')
similarity_results = compute_similarity_results(embeddings, docs)

print("\nRetrieval Results (CLS Pooling):")
for result in similarity_results:
    print(f"ID: {result['id']}")
    print(f"Rank: {result['rank']}")
    print(f"Doc Index: {result['doc_index']}")
    print(f"Text: {result['text']}")
    print(f"Score: {result['score']}")
    print(f"Percent Difference: {result['percent_difference']}%")
    print()

# --- Classification Task ---
# Sample dataset for binary sentiment classification
classification_data = [
    ("I love this movie!", 1),  # Positive
    ("This film is terrible.", 0),  # Negative
    ("Amazing performance by the cast!", 1),
    ("I hated the ending.", 0),
]

# Initialize classifier
num_labels = 2  # Binary classification
classifier = SentenceClassifier(base_model, num_labels).to(device)

# Training setup
optimizer = optim.Adam(classifier.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss()
num_epochs = 20

# Training loop
classifier.train()
for epoch in range(num_epochs):
    total_loss = 0
    for text, label in classification_data:
        inputs = tokenizer(text, padding=True,
                           return_tensors='pt', truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = torch.tensor([label], dtype=torch.long).to(device)

        optimizer.zero_grad()
        logits = classifier(inputs)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(
        f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(classification_data):.4f}")

# Save the trained classifier
model_save_path = "trained_sentence_classifier.pth"
torch.save(classifier.state_dict(), model_save_path)
print(f"\nTrained classifier saved to {model_save_path}")

# Load the trained classifier for inference
classifier = SentenceClassifier(base_model, num_labels).to(device)
classifier.load_state_dict(torch.load(model_save_path))
classifier.eval()

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
