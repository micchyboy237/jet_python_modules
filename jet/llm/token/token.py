from typing import Optional
from .token_image import calculate_img_tokens
import tiktoken


def token_counter(
    messages: Optional[list] = None,
    text: Optional[str] = None,
):
    num_tokens = 0  # Initialize num_tokens to avoid UnboundLocalError

    if text is None:
        if messages is not None:
            text = ""
            for message in messages:
                if message.get("content", None) is not None:
                    content = message.get("content")
                    if isinstance(content, str):
                        text += content
                    elif isinstance(content, list):
                        for c in content:
                            if c["type"] == "text":
                                text += c["text"]
                            elif c["type"] == "image_url":
                                if isinstance(c["image_url"], dict):
                                    image_url_dict = c["image_url"]
                                    detail = image_url_dict.get(
                                        "detail", "auto")
                                    url = image_url_dict.get("url")
                                    num_tokens += calculate_img_tokens(
                                        data=url, mode=detail
                                    )
                                elif isinstance(c["image_url"], str):
                                    image_url_str = c["image_url"]
                                    num_tokens += calculate_img_tokens(
                                        data=image_url_str, mode="auto"
                                    )
                if message.get("tool_calls"):
                    for tool_call in message["tool_calls"]:
                        if "function" in tool_call:
                            function_arguments = tool_call["function"]["arguments"]
                            text += function_arguments
        else:
            raise ValueError("text and messages cannot both be None")
    elif isinstance(text, list):
        text = "".join(t for t in text if isinstance(t, str))
    elif isinstance(text, str):
        pass

    encoding = tiktoken.get_encoding("cl100k_base")
    # Ensure all tokens are counted
    num_tokens += len(encoding.encode(text, disallowed_special=()))
    return num_tokens
