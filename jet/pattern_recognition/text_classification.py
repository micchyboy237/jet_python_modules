import logging
from typing import Dict, List, Tuple
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_and_preprocess_data(max_samples: int = 1000) -> Tuple[Dict, Dict]:
    """
    Load and preprocess IMDb dataset.

    Args:
        max_samples: Maximum samples per split to load.

    Returns:
        Tuple of (train_dataset, test_dataset).
    """
    logger.info("Loading IMDb dataset")
    dataset = load_dataset("imdb")

    # Limit samples for faster testing
    train_dataset = dataset["train"].select(
        range(min(max_samples, len(dataset["train"]))))
    test_dataset = dataset["test"].select(
        range(min(max_samples, len(dataset["test"]))))

    logger.info("Data preprocessing complete")
    return train_dataset, test_dataset


def tokenize_data(dataset: Dict, tokenizer: AutoTokenizer) -> Dict:
    """
    Tokenize dataset using BERT tokenizer.

    Args:
        dataset: Dataset to tokenize.
        tokenizer: BERT tokenizer.

    Returns:
        Tokenized dataset.
    """
    logger.info("Tokenizing dataset")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    return dataset.map(tokenize_function, batched=True)


def build_bert_model(model_name: str = "bert-base-uncased") -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    """
    Build BERT tokenizer and model.

    Args:
        model_name: Name of the BERT model.

    Returns:
        Tuple of (tokenizer, model).
    """
    logger.info("Building BERT model")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2)
    logger.info("Model and tokenizer loaded")
    return tokenizer, model


def train_model(model: AutoModelForSequenceClassification, train_dataset: Dict, test_dataset: Dict,
                output_dir: str = "./results") -> Trainer:
    """
    Train the BERT model.

    Args:
        model: BERT model to train.
        train_dataset: Tokenized training dataset.
        test_dataset: Tokenized test dataset.
        output_dir: Directory to save model outputs.

    Returns:
        Trainer object.
    """
    logger.info("Starting model training")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir="./logs",
        logging_steps=100,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, predictions)}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    logger.info("Model training complete")
    return trainer


def evaluate_model(trainer: Trainer) -> float:
    """
    Evaluate the model on test data.

    Args:
        trainer: Trained Trainer object.

    Returns:
        Test accuracy.
    """
    logger.info("Evaluating model")
    eval_results = trainer.evaluate()
    test_accuracy = eval_results["eval_accuracy"]
    logger.info(f"Test accuracy: {test_accuracy:.4f}")
    return test_accuracy


if __name__ == "__main__":
    # Load and preprocess data
    train_dataset, test_dataset = load_and_preprocess_data()

    # Build and tokenize
    tokenizer, model = build_bert_model()
    train_tokenized = tokenize_data(train_dataset, tokenizer)
    test_tokenized = tokenize_data(test_dataset, tokenizer)

    # Train and evaluate
    trainer = train_model(model, train_tokenized, test_tokenized)
    accuracy = evaluate_model(trainer)
