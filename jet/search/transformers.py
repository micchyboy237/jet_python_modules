import html


def unescape(text: str) -> str:
    decoded_text = text
    # Unescape &nbsp;
    decoded_text = decoded_text.replace("&nbsp;", " ")
    # Decode HTML entities
    decoded_text = html.unescape(decoded_text)
    # Directly return decoded text as utf-8 (no need for latin-1 encoding)
    return decoded_text


def decode_encoded_characters(text):
    decoded_lines = []  # To store processed lines
    for line in text.splitlines():
        line = unescape(line)
        # Replace curly quotes with standard quotes
        line = line.replace('“', '"').replace('”', '"')
        # Replace apostrophes with standard ones
        line = line.replace('’', "'").replace('‘', "'")
        decoded_lines.append(line)  # Append the processed line
    text = "\n".join(decoded_lines)

    # rules = [
    #     {'&#39;', '\''},
    #     {'&amp;', '&'},
    #     {'&lt;', '<'},
    #     {'&gt;', '>'},
    #     ('', '\''),
    #     ('，', ','),
    #     ('。', '.'),
    #     ('、', ','),
    #     ('”', '"'),
    #     ('“', '"'),
    #     ('∶', ':'),
    #     ('：', ':'),
    #     ('？', '?'),
    #     ('《', '"'),
    #     ('》', '"'),
    #     ('）', ')'),
    #     ('！', '!'),
    #     ('（', '('),
    #     ('；', ';'),
    #     ('１', '1'),
    #     ('」', '"'),
    #     ('「', '"'),
    #     ('０', '0'),
    #     ('３', '3'),
    #     ('２', '2'),
    #     ('５', '5'),
    #     ('６', '6'),
    #     ('９', '9'),
    #     ('７', '7'),
    #     ('８', '8'),
    #     ('４', '4'),
    #     ('．', '.'),
    #     ('～', '~'),
    #     ('’', '\''),
    #     ('‘', '\''),
    #     ('…', '...'),
    #     ('━', '-'),
    #     ('〈', '<'),
    #     ('〉', '>'),
    #     ('【', '['),
    #     ('】', ']'),
    #     ('％', '%')
    # ]

    # for (rule, replacement) in rules:
    #     text = text.replace(rule, replacement)

    return text


def clean_string(text: str) -> str:
    text = decode_encoded_characters(text)

    def remove_unmatched_characters(text, open_char, close_char):
        balance = 0
        new_text = ''
        for char in text:
            if char == open_char:
                balance += 1
                new_text += char
            elif char == close_char:
                if balance == 0:
                    continue  # Ignore this character as it's unmatched
                balance -= 1
                new_text += char
            else:
                new_text += char
        return new_text

    # Count the number of quotes
    quote_count = text.count('\"')

    # If the number of quotes is odd, remove the last one
    if quote_count % 2 != 0:
        last_quote_index = text.rfind('\"')
        text = text[:last_quote_index] + text[last_quote_index + 1:]

    # Process opening and closing characters
    for char_pair in [('(', ')'), ('[', ']'), ('{', '}')]:
        text = remove_unmatched_characters(text, *char_pair)
        text = remove_unmatched_characters(
            text[::-1], char_pair[1], char_pair[0])[::-1]

    # Remove leading/trailing commas
    text = text.strip(",")

    # Remove enclosing double quotes only if they're at both ends of the string
    if text.startswith("\"") and text.endswith("\""):
        text = text[1:-1]

    # Remove leading quote only if it's the only one in the string
    elif text.startswith("\"") and text.count('\"') == 1:
        text = text[1:]

    # Remove trailing quote only if it'text the only one in the string
    elif text.endswith("\"") and text.count('\"') == 1:
        text = text[:-1]

    # Remove enclosing single quotes only if they're at both ends of the string
    elif text.startswith("'") and text.endswith("'"):
        text = text[1:-1]

    # Remove leading single quote only if it's the only one in the string
    elif text.startswith("'") and text.count("'") == 1:
        text = text[1:]

    # Remove trailing single quote only if it'text the only one in the string
    elif text.endswith("'") and text.count("'") == 1:
        text = text[:-1]

    import re
    text = '\n'.join([line.strip() for line in text.split('\n')])
    # Reduce consecutive newlines to a single newline
    text = re.sub(r'\n{2,}', '\n', text)

    return text.strip()


if __name__ == '__main__':
    from jet.logger import logger
    decoded_query = decode_encoded_characters(
        "How many &amp;seasons and episodes does ”I’ll Become a Villainess Who Goes Down in History” anime&nbsp;have?")

    # Fix the assertion
    expected_output = "How many &seasons and episodes does \"I'll Become a Villainess Who Goes Down in History\" anime have?"
    assert decoded_query == expected_output, f"Test failed: {
        decoded_query} != {expected_output}"

    logger.success("PASSED!")
