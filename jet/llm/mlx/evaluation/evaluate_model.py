import json
import os
from pathlib import Path
import lm_eval
import mx.random
from importlib.metadata import version
from jet.llm.mlx.evaluation.argparse_utils import parse_evaluation_args
from mlxlm import MLXLM, chat_template_fn  # Assume MLXLM is in mlxlm.py
import argparse
import json
from pathlib import Path


def parse_evaluation_args(
    model: str = None,
    tasks: list = None,
    output_dir: str = ".",
    batch_size: int = 16,
    num_shots: int = None,
    max_tokens: int = None,
    limit: int = None,
    seed: int = 123,
    fewshot_as_multiturn: bool = False,
    apply_chat_template: bool = None,
    chat_template_args: str = "{}"
) -> argparse.Namespace:
    """
    Parse arguments for MLX model evaluation using lm-evaluation-harness.

    Args:
        model: Model to evaluate
        tasks: List of tasks to evaluate
        output_dir: Output directory for result files
        batch_size: Batch size for evaluation
        num_shots: Number of few-shot examples
        max_tokens: Maximum number of tokens to generate
        limit: Limit the number of examples per task
        seed: Random seed
        fewshot_as_multiturn: Whether to provide fewshot examples as multiturn conversation
        apply_chat_template: Whether to apply chat template to prompt
        chat_template_args: JSON string of arguments for tokenizer's apply_chat_template

    Returns:
        argparse.Namespace containing the parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Evaluate an MLX model using lm-evaluation-harness."
    )

    parser.add_argument("--model", help="Model to evaluate",
                        default=model, required=not model)
    parser.add_argument(
        "--tasks", nargs="+", help="Tasks to evaluate", default=tasks, required=not tasks)
    parser.add_argument(
        "--output-dir",
        default=output_dir,
        help="Output directory for result files."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=batch_size,
        help="Batch size"
    )
    parser.add_argument(
        "--num-shots",
        type=int,
        default=num_shots,
        help="Number of shots"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=max_tokens,
        help="Maximum number of tokens to generate. Defaults to the model's max context length."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=limit,
        help="Limit the number of examples per task."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=seed,
        help="Random seed."
    )
    parser.add_argument(
        "--fewshot-as-multiturn",
        action="store_true",
        help="Whether to provide the fewshot examples as a multiturn conversation or a single user turn.",
        default=fewshot_as_multiturn
    )
    parser.add_argument(
        "--apply-chat-template",
        action=argparse.BooleanOptionalAction,
        help="Specifies whether to apply a chat template to the prompt.",
        default=apply_chat_template
    )
    parser.add_argument(
        "--chat-template-args",
        type=json.loads,
        help="A JSON formatted string of arguments for the tokenizer's apply_chat_template",
        default=chat_template_args
    )

    return parser.parse_args()


def evaluate_model(args):

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Silence tokenizer warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Set random seed for reproducibility
    mx.random.seed(args.seed)

    # Initialize MLXLM model
    lm = MLXLM(
        args.model,
        max_tokens=args.max_tokens,
        use_chat_template=args.apply_chat_template,
    )
    MLXLM.apply_chat_template = chat_template_fn(**args.chat_template_args)

    # Run evaluation
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=args.tasks,
        fewshot_as_multiturn=args.fewshot_as_multiturn,
        apply_chat_template=lm.use_chat_template,
        num_fewshot=args.num_shots,
        limit=args.limit,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
    )

    # Save and print results
    file_keys = ["eval", args.model.replace("/", "_"), version("lm_eval")]
    if args.num_shots is not None:
        file_keys += [f"{args.num_shots:02d}"]
    file_keys += args.tasks
    filename = "_".join(file_keys)
    output_path = output_dir / f"{filename}.json"
    output_path.write_text(json.dumps(results["results"], indent=4))

    print("Evaluation Results:")
    for task, result in results["results"].items():
        print(f"\nTask: {task}")
        print(json.dumps(result, indent=4))


if __name__ == "__main__":
    evaluate_model()
