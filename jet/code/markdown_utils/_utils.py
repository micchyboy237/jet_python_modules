import re
from typing import List, Dict, Any
import os
from mrkdwn_analysis import MarkdownAnalyzer

def preprocess_custom_code_blocks(markdown_text: str) -> str:
    """
    Converts custom [code] ... [/code] blocks to standard fenced code blocks.
    
    Args:
        markdown_text: Raw Markdown string.
    
    Returns:
        Preprocessed Markdown string with converted code blocks.
    """

    def replace_match(match: re.Match) -> str:
        language = match.group(1) or "text"  # Default to 'text' if no language
        content = match.group(2).strip() if match.group(2) else ""
        result = f"```{language}\n{content}\n```"
        return result
    
    # Pattern: [code[:language]?] content [/code], or unclosed [code[:language]?]
    pattern = r'\[code(?:\:(\w+))?\]\s*(.*?)\s*(\[/code\]|\Z)'
    result = re.sub(pattern, replace_match, markdown_text, flags=re.DOTALL)
    
    return result

def extract_custom_code_blocks(file_path: str) -> List[Dict[str, Any]]:
    """
    Extracts code blocks after preprocessing custom delimiters.
    
    Args:
        file_path: Path to Markdown file.
    
    Returns:
        List of code block dicts.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    preprocessed = preprocess_custom_code_blocks(content)
    
    # Temporarily write preprocessed for analysis
    temp_file = "temp_preprocessed.md"
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(preprocessed)
    
    analyzer = MarkdownAnalyzer(temp_file)
    blocks = analyzer.identify_code_blocks().get("Code block", [])
    
    os.remove(temp_file)
    return blocks
