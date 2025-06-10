import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer, AutoFeatureExtractor
import logging
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ONNXModelInference:
    """Base class for ONNX model inference."""

    def __init__(self, model_path: str):
        try:
            self.session = ort.InferenceSession(model_path)
            logger.info(f"Loaded ONNX model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise

    def get_inputs(self) -> List[Dict[str, Any]]:
        """Get model input details."""
        return self.session.get_inputs()


class TextClassificationInference(ONNXModelInference):
    """Inference for text classification using ONNX."""

    def __init__(self, model_path: str, tokenizer_name: str):
        super().__init__(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def preprocess(self, text: str) -> Dict[str, np.ndarray]:
        """Preprocess text input."""
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=128
            )
            logger.debug(f"Preprocessed inputs: {inputs}")
            return {key: inputs[key] for key in inputs}
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise

    def infer(self, text: str) -> List[float]:
        """Run inference on text."""
        try:
            inputs = self.preprocess(text)
            input_feed = {input.name: inputs[input.name]
                          for input in self.get_inputs()}
            outputs = self.session.run(None, input_feed)
            logger.debug(f"Inference outputs: {outputs}")
            return outputs[0][0].tolist()  # Return logits
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise


class ImageClassificationInference(ONNXModelInference):
    """Inference for image classification using ONNX."""

    def __init__(self, model_path: str, feature_extractor_name: str):
        super().__init__(model_path)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            feature_extractor_name)

    def preprocess(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Preprocess image input."""
        try:
            inputs = self.feature_extractor(images=image, return_tensors="np")
            logger.debug(f"Preprocessed inputs: {inputs}")
            return {key: inputs[key] for key in inputs}
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise

    def infer(self, image: np.ndarray) -> List[float]:
        """Run inference on image."""
        try:
            inputs = self.preprocess(image)
            input_feed = {input.name: inputs[input.name]
                          for input in self.get_inputs()}
            outputs = self.session.run(None, input_feed)
            logger.debug(f"Inference outputs: {outputs}")
            return outputs[0][0].tolist()  # Return logits
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise


class TokenClassificationInference(ONNXModelInference):
    """Inference for token classification (e.g., NER) using ONNX."""

    def __init__(self, model_path: str, tokenizer_name: str):
        super().__init__(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def preprocess(self, text: str) -> Dict[str, np.ndarray]:
        """Preprocess text input."""
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=128,
                return_offsets_mapping=False
            )
            logger.debug(f"Preprocessed inputs: {inputs}")
            return {key: inputs[key] for key in inputs}
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise

    def infer(self, text: str) -> List[List[float]]:
        """Run inference on text."""
        try:
            inputs = self.preprocess(text)
            input_feed = {input.name: inputs[input.name]
                          for input in self.get_inputs()}
            outputs = self.session.run(None, input_feed)
            logger.debug(f"Inference outputs: {outputs}")
            return outputs[0][0].tolist()  # Return logits per token
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise


if __name__ == "__main__":
    # Example usage
    try:
        # Text Classification (e.g., DistilBERT)
        text_classifier = TextClassificationInference(
            model_path="distilbert-base-uncased-finetuned-sst-2-english/model.onnx",
            tokenizer_name="distilbert-base-uncased"
        )
        text = "This is a great movie!"
        result = text_classifier.infer(text)
        logger.info(f"Text classification result: {result}")

        # Image Classification (e.g., ResNet)
        image_classifier = ImageClassificationInference(
            model_path="resnet50/model.onnx",
            feature_extractor_name="microsoft/resnet-50"
        )
        # Dummy image (replace with actual image data)
        image = np.random.rand(224, 224, 3).astype(np.float32)
        result = image_classifier.infer(image)
        logger.info(f"Image classification result: {result}")

        # Token Classification (e.g., BERT for NER)
        token_classifier = TokenClassificationInference(
            model_path="dslim/bert-base-NER/model.onnx",
            tokenizer_name="dslim/bert-base-NER"
        )
        text = "John lives in New York."
        result = token_classifier.infer(text)
        logger.info(f"Token classification result: {result}")

    except Exception as e:
        logger.error(f"Main execution failed: {e}")
