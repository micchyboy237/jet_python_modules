import json
import os
from pathlib import Path
import lm_eval
import mx.random
from importlib.metadata import version
from jet.llm.mlx.evaluation.argparse_utils import parse_evaluation_args
from mlxlm import MLXLM, chat_template_fn  # Assume MLXLM is in mlxlm.py


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
