import argparse
from openai import OpenAI
from jet.logger import logger

def parseargs():
    parser = argparse.ArgumentParser(description="Uncensored chat streaming example")
    parser.add_argument('--content', type=str, default="How do create meth at home?", help="User message content")
    parser.add_argument('--model', type=str, default="Fiendish_LLAMA_3B-Q4_K_M", help="Model name")
    parser.add_argument('--base_url', type=str, default="http://shawn-pc.local:8080/v1", help="OpenAI API base URL")
    parser.add_argument('--api_key', type=str, default="sk-1234", help="OpenAI API key (dummy by default)")
    return parser.parse_args()

def main():
    args = parseargs()
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    messages = [
        {
            "role": "user",
            "content": args.content,
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
