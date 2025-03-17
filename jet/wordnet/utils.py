# from instruction_generator.wordnet.SpellingCorrectorNorvig import SpellingCorrectorNorvig
from typing import Any, Generator, Optional
from bs4 import BeautifulSoup
import bs4
from jet.utils.text import has_non_ascii
from jet.wordnet.sentence import split_sentences
from jet.wordnet.words import get_words
from lxml import etree
import re
import regex
import requests
from unidecode import unidecode


exclude = ["See also", "References", "Sources",
           "Further reading", "External links", "Mga Kawing Panlabas",
           "Silipin din", "Kawil panlabas", "Sipian",
           "Talasanggunian", "Mga sanggunian", "Mga kaugnay na naisulat",
           "Talababa",
           "Kawing Panlabas", "Panlabas na link", "Panlabas na kawing",
           "Ibang mapagbabasahan"]


def clean_texts(text):
    # This pattern matches nested parentheses as well
    text = regex.sub(r'\s*,?\s*(\((?:[^()]+|(?R))*\))', '', text)
    text = regex.sub(r'\s*,?\s*\[[^\]]*\]', '', text)
    text = regex.sub(r'[ ]+', ' ', text)
    text = "\n".join(line.strip() for line in text.split('\n'))
    # Ensure the text ends with a period if it originally did
    return text


def safe_int(value):
    """Safely convert value to integer, stripping non-numeric characters."""
    return int(''.join(filter(str.isdigit, value)))


def is_valid_element(element):
    """Check if an element should be parsed for data."""
    if isinstance(element, str):
        return False
    if element.name in ['style', 'script']:
        return False

    if element.name == "sup" and "reference" in element.attrs.get("class", []):
        return False

    if all(isinstance(child, bs4.Tag) and child.name == "a" for child in element.children):
        return False
    return True


def remove_excluded_elements(element):
    """Remove child elements that have excluded classes."""
    if not isinstance(element, bs4.Tag):
        return
    for excluded_class in ["mw-editsection", "infobox"]:
        for child in element.select(f".{excluded_class}"):
            child.decompose()


def strip_unwanted_tags(element):
    """Remove unwanted tags from an element and ensure it's valid."""

    for tag in element.find_all(['style', 'script', 'abbr']):
        tag.decompose()

    for sup in element.find_all('sup', class_='reference'):
        sup.decompose()

    return element


def clean_latex(latex_str):
    # Remove the \displaystyle command and other LaTeX formatting
    clean_str = re.sub(r'\\displaystyle', '', latex_str)

    # Replace \times with *
    clean_str = clean_str.replace(r'\t', '')
    clean_str = clean_str.replace(r'imes', '*')

    # Remove all whitespaces
    clean_str = re.sub(r'\s+', '', clean_str)

    # remove all "{" and "}"
    clean_str = clean_str.replace(r'{', '')
    clean_str = clean_str.replace(r'}', '')

    # add leading "{" and trailing "}"
    clean_str = '{' + clean_str + '}'

    # add single space around all non numeric and non operator characters
    clean_str = re.sub(r'([^0-9])', r' \1 ', clean_str)

    # replace multiple spaces with single space
    clean_str = re.sub(r'\s+', ' ', clean_str)

    # Further cleaning can be done here if needed
    return clean_str.strip()


def generate_readable_math_equations(html_string):
    # Parse the HTML content using lxml
    tree = etree.HTML(html_string)

    # Use XPath to extract the 'alttext' attribute of the 'math' element
    alttexts = tree.xpath('//math/@alttext')

    # Clean and prepare the equations for the text corpus
    equations_for_corpus = [clean_latex(alt) for alt in alttexts]

    # Join the equations into a single string separated by new lines
    corpus_text = '\n'.join(equations_for_corpus)

    return corpus_text


def process_elements(elements):
    stack = []
    root_section = {"title": None, "content": []}
    stack.append(root_section)
    prev_element_type = None

    i = 0
    while i < len(elements):
        elem = elements[i]

        if not is_valid_element(elem):
            i += 1
            continue

        remove_excluded_elements(elem)

        if elem.name in ['h2', 'h3', 'h4']:
            heading_text = strip_unwanted_tags(elem).get_text().strip()

            if heading_text in exclude:

                while True:
                    i += 1
                    if i >= len(elements):
                        break
                    next_elem = elements[i]

                    if next_elem.name in ['h2', 'h3', 'h4']:
                        break
                continue

            current_level = int(elem.name[1])

            while len(stack) > current_level - 1:
                stack.pop()

            new_section = {"title": heading_text, "content": []}
            stack[-1]["content"].append(new_section)
            stack.append(new_section)
            prev_element_type = None

        elif elem.name == 'p':
            stripped_elem = strip_unwanted_tags(elem)
            # join stripped_elem.contents
            contents = []
            for child in stripped_elem.children:
                child_text = ''

                if isinstance(child, bs4.Tag):
                    # if child is <math> or it has a <math> child
                    if child.name == 'math' or child.find('math'):
                        # get math element
                        math_elem = child if child.name == 'math' else child.find(
                            'math')
                        # get math element string
                        math_elem_str = str(math_elem)
                        child_text = generate_readable_math_equations(
                            math_elem_str)
                    else:
                        child_text = child.get_text()

                else:
                    child_text = str(child)

                # strip right newlines
                child_text = child_text.rstrip('\n')

                # add space if contents[-1] is empty
                if contents and not contents[-1]:
                    child_text = f" {child_text}"

                contents.append(child_text)

            description_text = ''.join(contents).strip()
            description_text = clean_texts(description_text)
            description_text = preserve_tagalog_chars(description_text)

            next_elem = elements[i + 1] if i + 1 < len(elements) else None
            if next_elem and next_elem.name in ['ul', 'table']:
                content_group = {"content": []}
                if description_text:
                    content_group["description"] = description_text
                stack[-1]['content'].append(content_group)
                stack.append(content_group)
                prev_element_type = None
            else:
                if description_text:
                    stack[-1]['content'].append(
                        {"description": description_text})
                prev_element_type = 'p'

        elif elem.name == 'ul':
            if prev_element_type != 'ul':
                current_list = []
                stack[-1]['content'].append({"list": current_list})
            for li in elem.find_all('li'):
                stripped_li = strip_unwanted_tags(li)
                current_list.append(stripped_li.get_text().strip())
            prev_element_type = 'ul'

            if isinstance(stack[-1], dict) and "description" in stack[-1] and stack[-1]["description"]:
                stack.pop()

        elif elem.name == 'table':
            if prev_element_type != 'table':
                table_data = []
                stack[-1]['content'].append({"table": table_data})
            for row in elem.find_all('tr'):
                row_data = []
                for cell in row.find_all(['th', 'td']):
                    stripped_cell = strip_unwanted_tags(cell)
                    cell_data = {
                        'type': 'heading' if cell.name == 'th' else 'data',
                        'colspan': safe_int(cell['colspan']) if cell.has_attr('colspan') else 1,
                        'description': stripped_cell.get_text().strip()
                    }
                    if cell.has_attr('scope'):
                        cell_data['scope'] = cell['scope']
                    row_data.append(cell_data)
                table_data.append(row_data)
            prev_element_type = 'table'

            if isinstance(stack[-1], dict) and "description" in stack[-1] and stack[-1]["description"]:
                stack.pop()

        i += 1

    return root_section["content"]


def is_valid_path(path):
    # Check for the presence of a colon
    if ":" in path:
        return None

    return True


def autocorrect_texts(texts):
    text_corrector = SpellingCorrectorNorvig()

    results_stream = text_corrector.autocorrect_texts(texts)
    results = []
    for result in results_stream:
        results.append(result)

        yield result

    return results


def preserve_tagalog_chars(s: str) -> str:
    special_char_mapping = {
        "ñ": "<ntilde>",
        "Ñ": "<NTILDE>",

    }

    for char, placeholder in special_char_mapping.items():
        s = s.replace(char, placeholder)

    s = unidecode(s)

    for char, placeholder in special_char_mapping.items():
        s = s.replace(placeholder, char)

    return s


def get_content_from_url(url):
    """
    Get the content and formatted content string from request text html.
    """
    wiki_path = url.split("/wiki/")[1]
    if not is_valid_wiki_path(wiki_path):
        return None

    req = requests.get(url)
    soup = BeautifulSoup(req.text, "lxml")

    page_title = soup.find("h1", class_="firstHeading").text
    page_url = soup.find("link", rel="canonical")["href"]

    content_div = soup.select_one("#mw-content-text .mw-parser-output")
    content = []
    content_str = ""

    if content_div:
        meta_page_prop_elm = content_div.select_one(
            "meta[property='mw:PageProp/toc']")

        if meta_page_prop_elm:
            prev_p_tags = meta_page_prop_elm.find_all_previous("p")
            prev_p_tags = [
                p_tag for p_tag in prev_p_tags if p_tag.parent == content_div]
            prev_p_tags.reverse()
            p_tags = prev_p_tags
        else:
            p_tags = content_div.find_all("p")
            # Remove the last item in p_tags if its first child contains .mw-file-element
            if p_tags and p_tags[-1].find("img", class_="mw-file-element"):
                p_tags = p_tags[:-1]

        # Assuming process_elements is defined elsewhere
        processed_tags = process_elements(p_tags)
        content.extend(processed_tags)
        for content_piece in content:
            content_str += content_piece.get("description", "") + "\n"
        content_str = content_str.strip()

    if len(content_str.split()) <= 60 or has_non_ascii(content_str):
        return None

    return {
        "title": page_title,
        "content": content_str,
        "url": page_url
    }


def sliding_window(text: str | list[Any], window_size: int, step_size: int = 1) -> Generator[list[str], None, None]:
    """
    Generate sequences using the sliding window approach.

    :param text: A list of tokens (words or characters) from the text corpus.
    :param window_size: The size of the window (number of tokens in each sample).
    :param step_size: The number of tokens to move the window at each step.
    :return: A generator of text sequences.
    """
    if isinstance(text, str):
        text = [word for sentence in split_sentences(
            text) for word in get_words(sentence)]

    data = text.copy()
    last_index = 0
    # Iterate over the data, generating windows of the specified size and step
    for i in range(0, len(data) - window_size + 1, step_size):
        last_index = i + window_size
        # Yield the current window
        yield data[i:last_index]

    # yield remaining data if any
    remaining_data = data[last_index:]
    if remaining_data:
        yield data[last_index:]


def increasing_window(text: str | list[Any], step_size: int = 1, max_window_size: Optional[int] = None):
    if isinstance(text, str):
        text = [word for sentence in split_sentences(
            text) for word in get_words(sentence)]

    if not max_window_size or max_window_size > len(text):
        max_window_size = len(text)

    data = text.copy()
    last_index = 1

    for i in range(1, max_window_size + 1, step_size):
        last_index = i
        yield data[0:i]

    # yield remaining data if any
    remaining_data = data[last_index:]
    if remaining_data:
        yield from increasing_window(remaining_data, step_size, max_window_size)


if __name__ == "__main__":
    text_corpus = "The quick brown fox jumps over the lazy dog. This is a simple text example for illustration."
    window_size = 3  # Number of tokens in each window
    step_size = 1    # Move the window by one token each time

    # Generate and print the sequences
    result = list(sliding_window(text_corpus, window_size, step_size))
    for sequence in sliding_window(text_corpus, window_size, step_size):
        print(sequence)

    # Generate and print the sequences
    result = list(increasing_window(text_corpus, step_size))
    for sequence in increasing_window(text_corpus, step_size=step_size):
        print(sequence)
