import argparse
import asyncio
import logging

from jet.adapters.llama_cpp.config import LLM_BASE_URL, LLM_MODEL
from jet_telemetry import get_trace_url, initialize_telemetry

from .agent import run_agent
from .config import (
    DEFAULT_LOG_LEVEL,
    DEFAULT_MAX_REPEATS,
    DEFAULT_MAX_STEPS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MCP_SERVER_URL,
    DEFAULT_PRESENCE_PENALTY,
    DEFAULT_START_URL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_TRACE_NAME,
    setup_logging,
)

logger = logging.getLogger("web_surfer_agent")


def get_args():
    """Parse command-line arguments with 1-2 char shorthands for keyword args."""
    parser = argparse.ArgumentParser(description="Run the Web Surfer MCP Agent.")
    parser.add_argument(
        "user_goal",
        nargs="?",
        default="Add a todo item 'Buy Milk' and then mark it as done.",
        help="Goal for the agent to accomplish (default: %(default)s)",
    )
    parser.add_argument(
        "-u",
        "--url",
        dest="start_url",
        default=DEFAULT_START_URL,
        help="Starting URL for the browser (default: %(default)s)",
    )
    parser.add_argument(
        "-ms",
        "--mcp-server",
        dest="mcp_server_url",
        default=DEFAULT_MCP_SERVER_URL,
        help=f"MCP server URL (default: {DEFAULT_MCP_SERVER_URL})",
    )
    parser.add_argument(
        "-la",
        "--llama-api",
        dest="llama_api_url",
        default=f"{LLM_BASE_URL}/chat/completions",
        help=f"LLaMA API URL (default: {LLM_BASE_URL}/chat/completions)",
    )
    parser.add_argument(
        "-lm",
        "--llama-model",
        dest="llama_model",
        default=LLM_MODEL,
        help=f"LLaMA model name (default: {LLM_MODEL})",
    )
    parser.add_argument(
        "-s",
        "--max-steps",
        dest="max_steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"Maximum number of steps (default: {DEFAULT_MAX_STEPS})",
    )
    parser.add_argument(
        "-r",
        "--max-repeats",
        dest="max_repeats",
        type=int,
        default=DEFAULT_MAX_REPEATS,
        help=f"Maximum repeated actions before stopping (default: {DEFAULT_MAX_REPEATS})",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        dest="temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="LLM temperature for response generation (default: 0.3)",
    )
    parser.add_argument(
        "-mt",
        "--max-tokens",
        dest="max_tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum tokens for LLM response (default: 500)",
    )
    parser.add_argument(
        "-tp",
        "--top-p",
        dest="top_p",
        type=float,
        default=DEFAULT_TOP_P,
        help="Top-p sampling parameter (default: 0.95)",
    )
    parser.add_argument(
        "-pp",
        "--presence-penalty",
        dest="presence_penalty",
        type=float,
        default=DEFAULT_PRESENCE_PENALTY,
        help="Presence penalty for LLM (default: 1.5)",
    )
    parser.add_argument(
        "-et",
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        default=False,
        help="Enable thinking mode in chat template (default: False)",
    )
    parser.add_argument(
        "-ll",
        "--log-level",
        dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=DEFAULT_LOG_LEVEL,
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "-tn",
        "--trace-name",
        dest="trace_name",
        default=DEFAULT_TRACE_NAME,
        help="Trace/project name for telemetry (default: %(default)s)",
    )
    return parser.parse_args()


def main():
    args = get_args()
    setup_logging(args.log_level)
    initialize_telemetry(service_name=args.trace_name)

    try:
        asyncio.run(run_agent(args.user_goal, args))
    except KeyboardInterrupt:
        logger.info("\n⚠️ Interrupted by user")
    except Exception as e:
        logger.error("💥 Fatal error: %s", e)
        raise
    finally:
        trace_url = get_trace_url()
        if trace_url:
            print(f"\n📊 View trace in Phoenix: {trace_url}")
        else:
            print("\n⚠️ No trace URL available (telemetry may not be initialized)")


if __name__ == "__main__":
    main()
