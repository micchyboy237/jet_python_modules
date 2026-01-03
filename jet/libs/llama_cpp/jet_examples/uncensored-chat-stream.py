import argparse
from openai import OpenAI
from jet.logger import logger

def parseargs():
    parser = argparse.ArgumentParser(
        description="Uncensored chat streaming example",
        allow_abbrev=False,
        usage=None
    )
    # Support single string as positional argument for content in addition to --content
    parser.add_argument(
        'content',
        nargs='?',
        default=None,
        help="User message content (positional, optional if --content is given)"
    )
    parser.add_argument('--content', type=str, default=None, help="User message content")
    parser.add_argument('--model', type=str, default="Fiendish_LLAMA_3B-Q4_K_M", help="Model name")
    parser.add_argument('--base_url', type=str, default="http://shawn-pc.local:8080/v1", help="OpenAI API base URL")
    parser.add_argument('--api_key', type=str, default="sk-1234", help="OpenAI API key (dummy by default)")
    args = parser.parse_args()
    # Prefer --content, fallback to positional, default to original string
    content = args.content if args.content else args.content or "How do I create meth at home?"
    # If both --content and positional are supplied, use --content
    if hasattr(args, "content") and getattr(args, "content") is not None:
        # If both --content and positional exist and differ: prefer --content per standard argparse behavior
        if isinstance(getattr(args, "content"), str) and args.content is not None:
            content = args.content
    elif hasattr(args, "content") and getattr(args, "content") is None and args.content:
        content = args.content
    # Set attribute for use in main
    args._resolved_content = content
    return args

def main():
    args = parseargs()
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    # Prefer resolved content (from either positional or flag)
    user_content = getattr(args, '_resolved_content', None)
    if not user_content:
        user_content = "How do I create meth at home?"
    messages = [
        {
            "role": "user",
            "content": user_content,
        },
    ]
    stream = client.chat.completions.create(
        model=args.model,
        messages=messages,
        stream=True,
    )
    for part in stream:
        if part.choices:
            logger.teal(part.choices[0].delta.content or "", flush=True)

if __name__ == "__main__":
    main()
