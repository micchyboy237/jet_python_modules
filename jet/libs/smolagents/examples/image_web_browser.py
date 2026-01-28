import argparse

from jet.libs.smolagents.docs.web_browser import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run browser agent with Helium + LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script.py                                      # default task, visible browser
  python script.py --headless                            # default task, headless
  python script.py -H                                    # same as --headless
  python script.py "Go to google.com and search for xAI" # custom task, visible
  python script.py -t "Visit example.com" -H             # custom task + headless
  python script.py -t "Visit x.ai" --headless            # custom task + headless
        """,
    )

    # Task – positional or -t / --task
    parser.add_argument(
        "task_pos",
        nargs="?",
        default=None,
        help="The task for the agent (positional – optional)",
    )
    parser.add_argument(
        "-t",
        "--task",
        dest="task_opt",
        default=None,
        help="The task for the agent (alternative to positional)",
    )

    parser.add_argument(
        "-H",
        "--headless",
        action="store_true",
        help="Run browser in headless mode (no visible window)",
    )

    args = parser.parse_args()

    # Resolve task: prefer -t/--task if given, otherwise use positional
    chosen_task = args.task_opt if args.task_opt is not None else args.task_pos

    main(headless=args.headless, task=chosen_task)
