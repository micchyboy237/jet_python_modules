# Option 1 – The safe & most popular way
import re
from typing import Optional
import yaml
import json

def extract_code_block_content(text: str, lang: Optional[str] = None) -> str:
    if lang:
        search_string = f"```{lang}"
    else:
        search_string = "```"

    start = text.find(search_string)
    if start == -1:
        return text

    start += len(search_string)

    # Skip optional language identifier + possible spaces/newline when lang=None
    if not lang:
        # Find the end of the opening fence line
        newline_pos = text.find("\n", start)
        if newline_pos != -1:
            start = newline_pos + 1
        # else: malformed, but we'll just take from here

    end = text.find("```", start)
    return text[start:end].strip() if end != -1 else text[start:].strip()

def yaml_to_dict(yaml_str: str) -> dict:
    """
    Convert YAML string to dict safely.
    """
    data = yaml.safe_load(yaml_str)
    return data

def yaml_to_json(yaml_str: str, indent: int = 2) -> str:
    """
    Convert YAML string to formatted JSON string safely.
    """
    data = yaml.safe_load(yaml_str)
    return json.dumps(data, indent=indent, ensure_ascii=False)

def yaml_file_to_json_file(input_path: str, output_path: str) -> None:
    with open(input_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def remove_markdown_links(text: str, remove_text: bool = False) -> str:
    """
    Remove all markdown links and optionally their visible text content.
    
    Args:
        text: Markdown text possibly containing links or images.
        remove_text: If True, removes both the markdown link syntax and its text/alt label.
                     If False (default), keeps the label text.
    
    Returns:
        str: Text with markdown links removed or replaced by labels depending on remove_text flag.
    """
    pattern = re.compile(
        r'(!)?\[([^\[\]]*?(?:\[[^\[\]]*?\])*?[^\[\]]*?)\]\((\S+?)(?:\s+"([^"]+)")?\)'
    )
    output = ""
    last_end = 0

    for match in pattern.finditer(text):
        start, end = match.span()
        label = match.group(2).strip()
        output += text[last_end:start]

        if not remove_text:
            output += label
        # if remove_text=True, skip adding label text entirely

        last_end = end

    output += text[last_end:]
    return output.strip()
