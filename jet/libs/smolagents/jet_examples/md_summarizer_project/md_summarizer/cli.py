"""Command-line entry point.

Usage:
    python -m md_summarizer.cli /path/to/docs
    python -m md_summarizer.cli /path/to/docs --server-url http://localhost:8080
    python -m md_summarizer.cli --demo
        (runs against the bundled demo_docs/ with a mock LLM -- no server needed)
"""

import argparse
import logging
import sys
from pathlib import Path

from .config import PipelineConfig
from .llm_client import LlamaCppClient, LLMRequestError, MockLLMClient
from .pipeline import run_pipeline


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recursively summarize a directory of markdown docs with a small local llama.cpp model."
    )
    parser.add_argument(
        "target_dir", nargs="?", type=Path, default=None,
        help="Directory to scan recursively for .md files.",
    )
    parser.add_argument(
        "--server-url", default="http://localhost:8080",
        help="Base URL of the running llama-server instance (default: %(default)s).",
    )
    parser.add_argument(
        "--model-ctx", type=int, default=10_000,
        help="Total context window of the model in tokens, matching llama-server's -c flag (default: %(default)s).",
    )
    parser.add_argument(
        "--reserved-output", type=int, default=700,
        help="Tokens reserved for the model's output per call (default: %(default)s).",
    )
    parser.add_argument(
        "--prompt-overhead", type=int, default=400,
        help="Tokens reserved for system prompt + formatting per call (default: %(default)s).",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--no-verify", action="store_true", help="Skip the verifier pass.")
    parser.add_argument("--verify-sample-size", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Where to write the final digest + report (default: prints to stdout only).",
    )
    parser.add_argument("--log-file", type=Path, default=None, help="Optional path to also write logs to a file.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--demo", action="store_true",
        help="Run against the bundled demo_docs/ using a mock LLM client -- no server required.",
    )
    return parser


def configure_logging(verbose: bool, log_file: Path = None) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging(args.verbose, args.log_file)
    logger = logging.getLogger("md_summarizer.cli")

    if args.demo:
        target_dir = Path(__file__).resolve().parent.parent / "demo_docs"
        llm = MockLLMClient()
        logger.info("running in --demo mode against %s with a mock LLM client (no server needed)", target_dir)
    else:
        if args.target_dir is None:
            logger.error("target_dir is required unless --demo is passed")
            return 2
        target_dir = args.target_dir
        llm = LlamaCppClient(args.server_url)

    config = PipelineConfig(
        model_ctx_tokens=args.model_ctx,
        reserved_output_tokens=args.reserved_output,
        system_prompt_overhead_tokens=args.prompt_overhead,
        temperature=args.temperature,
        verify_sample_size=args.verify_sample_size,
    )
    logger.info(
        "token budget per call: %d input tokens (ctx=%d - output=%d - overhead=%d)",
        config.input_token_budget, config.model_ctx_tokens,
        config.reserved_output_tokens, config.system_prompt_overhead_tokens,
    )

    try:
        final_digest, verification_report, _tree = run_pipeline(
            target_dir, llm, config, run_verification=not args.no_verify
        )
    except (LLMRequestError, ValueError) as exc:
        logger.error(str(exc))
        return 1

    report_lines = ["# Final digest", "", final_digest, ""]
    if verification_report is not None:
        report_lines += ["# Verification spot-check", "", verification_report, ""]
    report = "\n".join(report_lines)

    print("\n" + "=" * 70)
    print(report)

    if args.output:
        args.output.write_text(report, encoding="utf-8")
        logger.info("wrote final report to %s", args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
