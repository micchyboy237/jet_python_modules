import pytest
from jet.llm.mlx.evaluation.argparse_utils import parse_evaluation_args


class TestParseEvaluationArgs:
    def test_default_args(self):
        # Test basic required arguments
        expected = {
            'model': 'mistral-7b',
            'tasks': ['hellaswag'],
            'output_dir': './results',
            'batch_size': 16,
            'num_shots': None,
            'max_tokens': None,
            'limit': None,
            'seed': 123,
            'fewshot_as_multiturn': False,
            'apply_chat_template': None,
            'chat_template_args': {}
        }
        result = vars(parse_evaluation_args(model='mistral-7b',
                      tasks=['hellaswag'], output_dir='./results'))
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_full_args(self):
        # Test full configuration
        expected = {
            'model': 'llama-3b',
            'tasks': ['mmlu', 'gsm8k'],
            'output_dir': './eval_results',
            'batch_size': 32,
            'num_shots': 5,
            'max_tokens': 2048,
            'limit': 100,
            'seed': 42,
            'fewshot_as_multiturn': True,
            'apply_chat_template': True,
            'chat_template_args': {'enable_thinking': True}
        }
        result = vars(parse_evaluation_args(
            model='llama-3b',
            tasks=['mmlu', 'gsm8k'],
            output_dir='./eval_results',
            batch_size=32,
            num_shots=5,
            max_tokens=2048,
            limit=100,
            seed=42,
            fewshot_as_multiturn=True,
            apply_chat_template=True,
            chat_template_args='{"enable_thinking": true}'
        ))
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_invalid_chat_template_args(self):
        # Test invalid JSON string for chat_template_args
        with pytest.raises(json.JSONDecodeError):
            parse_evaluation_args(
                model='mistral-7b',
                tasks=['hellaswag'],
                chat_template_args='invalid_json'
            )
