import html


def unescape(text: str) -> str:
    decoded_text = text
    # Unescape &nbsp;
    decoded_text = decoded_text.replace("&nbsp;", " ")
    # Decode HTML entities
    decoded_text = html.unescape(decoded_text)
    # Directly return decoded text as utf-8 (no need for latin-1 encoding)
    return decoded_text


def decode_encoded_characters(text: str) -> str:
    decoded_lines = []  # To store processed lines
    for line in text.splitlines():
        line = unescape(line)
        # Replace curly quotes with standard quotes
        line = line.replace('“', '"').replace('”', '"')
        # Replace apostrophes with standard ones
        line = line.replace('’', "'").replace('‘', "'")
        decoded_lines.append(line)  # Append the processed line
    return "\n".join(decoded_lines)


if __name__ == '__main__':
    from jet.logger import logger
    decoded_query = decode_encoded_characters(
        "How many &amp;seasons and episodes does ”I’ll Become a Villainess Who Goes Down in History” anime&nbsp;have?")

    # Fix the assertion
    expected_output = "How many &seasons and episodes does \"I'll Become a Villainess Who Goes Down in History\" anime have?"
    assert decoded_query == expected_output, f"Test failed: {
        decoded_query} != {expected_output}"

    logger.success("PASSED!")
