import logging
from typing import Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_and_preprocess_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess CIFAR-10 dataset.

    Returns:
        Tuple of (train_images, train_labels, test_images, test_labels).
    """
    logger.info("Loading CIFAR-10 dataset")
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    # Normalize pixel values to [0, 1]
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    logger.info("Data preprocessing complete")
    return train_images, train_labels, test_images, test_labels


def build_cnn_model(input_shape: Tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    """
    Build a CNN model for image classification.

    Args:
        input_shape: Shape of input images (height, width, channels).
        num_classes: Number of output classes.

    Returns:
        Compiled Keras model.
    """
    logger.info("Building CNN model")
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    logger.info("Model compilation complete")
    return model


def train_model(model: tf.keras.Model, train_data: Tuple[np.ndarray, np.ndarray],
                test_data: Tuple[np.ndarray, np.ndarray], epochs: int = 10) -> tf.keras.callbacks.History:
    """
    Train the CNN model.

    Args:
        model: Keras model to train.
        train_data: Tuple of (train_images, train_labels).
        test_data: Tuple of (test_images, test_labels).
        epochs: Number of training epochs.

    Returns:
        Training history.
    """
    logger.info("Starting model training")
    train_images, train_labels = train_data
    test_images, test_labels = test_data

    history = model.fit(train_images, train_labels, epochs=epochs,
                        validation_data=(test_images, test_labels),
                        batch_size=64, verbose=1)
    logger.info("Model training complete")
    return history


def evaluate_model(model: tf.keras.Model, test_images: np.ndarray, test_labels: np.ndarray) -> float:
    """
    Evaluate the model on test data.

    Args:
        model: Trained Keras model.
        test_images: Test images.
        test_labels: Test labels.

    Returns:
        Test accuracy.
    """
    logger.info("Evaluating model")
    _, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    logger.info(f"Test accuracy: {test_accuracy:.4f}")
    return test_accuracy


if __name__ == "__main__":
    # Load and preprocess data
    train_images, train_labels, test_images, test_labels = load_and_preprocess_data()

    # Build and train model
    model = build_cnn_model(input_shape=(32, 32, 3), num_classes=10)
    history = train_model(
        model, (train_images, train_labels), (test_images, test_labels))

    # Evaluate model
    accuracy = evaluate_model(model, test_images, test_labels)
