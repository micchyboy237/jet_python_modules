from jet.code.markdown_code_extractor import MarkdownCodeExtractor


def extract_block_content(text: str) -> str:
    extractor = MarkdownCodeExtractor()
    code_blocks = extractor.extract_code_blocks(text)
    return code_blocks[0]["code"]


def extract_json_block_content(text: str) -> str:
    start = text.find("```json")
    if start == -1:
        return text

    start += len("```json")
    end = text.find("```", start)
    return text[start:end].strip() if end != -1 else text[start:].strip()
