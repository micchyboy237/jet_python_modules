import pytest
from unittest.mock import MagicMock
from jet.llm.mlx.utils.logit_bias import convert_logit_bias


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.encode.side_effect = lambda x, **kwargs: [
        100] if x == "valid" else []
    return tokenizer


def test_convert_logit_bias_none_input(mock_tokenizer):
    expected = None
    result = convert_logit_bias(None, mock_tokenizer)
    assert result == expected, "None input should return None"


def test_convert_logit_bias_dict_input(mock_tokenizer):
    input_dict = {100: 10.0}
    expected = input_dict
    result = convert_logit_bias(input_dict, mock_tokenizer)
    assert result == expected, "Dictionary input should return unchanged"


def test_convert_logit_bias_string_input(mock_tokenizer):
    expected = {100: 15.0}
    result = convert_logit_bias("valid", mock_tokenizer)
    assert result == expected, "String input should convert to token ID dict"


def test_convert_logit_bias_list_input(mock_tokenizer):
    expected = {100: 15.0}
    result = convert_logit_bias(["valid"], mock_tokenizer)
    assert result == expected, "List input should convert to token ID dict"


def test_convert_logit_bias_invalid_input(mock_tokenizer):
    expected = None
    result = convert_logit_bias(["invalid"], mock_tokenizer)
    assert result == expected, "Invalid input should return None"


def test_convert_logit_bias_mixed_input(mock_tokenizer):
    expected = {100: 15.0}
    result = convert_logit_bias(["valid", "invalid"], mock_tokenizer)
    assert result == expected, "Mixed input should process valid tokens only"
