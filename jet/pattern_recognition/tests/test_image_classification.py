import pytest
import numpy as np
from jet.pattern_recognition.image_classification import load_and_preprocess_data, build_cnn_model


class TestImageClassification:
    def test_load_and_preprocess_data(self):
        """
        Test data loading and preprocessing.
        """
        train_images, train_labels, test_images, test_labels = load_and_preprocess_data()

        expected_train_shape = (50000, 32, 32, 3)
        expected_test_shape = (10000, 32, 32, 3)
        expected_max_pixel = 1.0
        expected_min_pixel = 0.0

        result_train_shape = train_images.shape
        result_test_shape = test_images.shape
        result_max_pixel = np.max(train_images)
        result_min_pixel = np.min(train_images)

        assert result_train_shape == expected_train_shape, f"Expected {expected_train_shape}, got {result_train_shape}"
        assert result_test_shape == expected_test_shape, f"Expected {expected_test_shape}, got {result_test_shape}"
        assert result_max_pixel <= expected_max_pixel, f"Expected max pixel <= {expected_max_pixel}, got {result_max_pixel}"
        assert result_min_pixel >= expected_min_pixel, f"Expected min pixel >= {expected_min_pixel}, got {result_min_pixel}"

    def test_build_cnn_model(self):
        """
        Test CNN model construction.
        """
        input_shape = (32, 32, 3)
        num_classes = 10
        model = build_cnn_model(input_shape, num_classes)

        expected_layers = 8  # Conv2D, MaxPool, Conv2D, MaxPool, Conv2D, Flatten, Dense, Dense
        expected_output_shape = (None, num_classes)

        result_layers = len(model.layers)
        result_output_shape = model.output_shape

        assert result_layers == expected_layers, f"Expected {expected_layers} layers, got {result_layers}"
        assert result_output_shape == expected_output_shape, f"Expected {expected_output_shape}, got {result_output_shape}"
