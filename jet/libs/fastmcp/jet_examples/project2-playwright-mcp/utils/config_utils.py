# Option 1 – The safe & most popular way
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
