import pytest
from jet.pattern_recognition.text_classification import load_and_preprocess_data, build_bert_model, tokenize_data
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class TestTextClassification:
    def test_load_and_preprocess_data(self):
        """
        Test data loading and preprocessing.
        """
        train_dataset, test_dataset = load_and_preprocess_data(max_samples=100)

        expected_train_size = 100
        expected_test_size = 100
        expected_keys = ["text", "label"]

        result_train_size = len(train_dataset)
        result_test_size = len(test_dataset)
        result_keys = list(train_dataset[0].keys())

        assert result_train_size == expected_train_size, f"Expected {expected_train_size}, got {result_train_size}"
        assert result_test_size == expected_test_size, f"Expected {expected_test_size}, got {result_test_size}"
        assert result_keys == expected_keys, f"Expected {expected_keys}, got {result_keys}"

    def test_build_bert_model(self):
        """
        Test BERT model and tokenizer construction.
        """
        tokenizer, model = build_bert_model()

        expected_tokenizer_type = AutoTokenizer
        expected_model_type = AutoModelForSequenceClassification
        expected_num_labels = 2

        result_tokenizer_type = type(tokenizer)
        result_model_type = type(model)
        result_num_labels = model.config.num_labels

        assert result_tokenizer_type == expected_tokenizer_type, f"Expected {expected_tokenizer_type}, got {result_tokenizer_type}"
        assert result_model_type == expected_model_type, f"Expected {expected_model_type}, got {result_model_type}"
        assert result_num_labels == expected_num_labels, f"Expected {expected_num_labels}, got {result_num_labels}"

    def test_tokenize_data(self):
        """
        Test data tokenization.
        """
        tokenizer, _ = build_bert_model()
        train_dataset, _ = load_and_preprocess_data(max_samples=2)
        tokenized_dataset = tokenize_data(train_dataset, tokenizer)

        expected_keys = ["input_ids", "token_type_ids",
                         "attention_mask", "label"]
        expected_max_length = 128

        result_keys = list(tokenized_dataset[0].keys())
        result_input_ids_length = len(tokenized_dataset[0]["input_ids"])

        assert result_keys == expected_keys, f"Expected {expected_keys}, got {result_keys}"
        assert result_input_ids_length == expected_max_length, f"Expected {expected_max_length}, got {result_input_ids_length}"
