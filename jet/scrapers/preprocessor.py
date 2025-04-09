import json
import html2text
import parsel
import math
from .utils import clean_newlines, clean_tags, clean_text
from jet.logger import logger


def convert_html_to_markdown(html_string):
    # Use the html2text library to convert HTML to Markdown
    converter = html2text.HTML2Text()

    # Configure the converter if necessary
    converter.ignore_links = True
    converter.ignore_images = True
    converter.ignore_emphasis = True
    converter.mark_code = True
    # converter.bypass_tables = True
    converter.body_width = 0  # Prevent line wrapping

    # Convert the HTML string to Markdown
    markdown_string = converter.handle(html_string)

    return markdown_string.strip()


def get_header_level(header: str) -> int:
    """Get the header level of a markdown header or HTML header tag."""
    if header.startswith("#"):
        header_level = 0
        for c in header:
            if c == "#":
                header_level += 1
            else:
                break
        return header_level
    elif header.startswith("h") and header[1].isdigit() and 1 <= int(header[1]) <= 6:
        return int(header[1])
    else:
        raise ValueError(f"Invalid header format: {header}")


def get_header_contents(md_text: str, headers_to_split_on: list[tuple[str, str]] = []) -> list[dict]:
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on, strip_headers=False, return_each_line=False)
    md_header_splits = markdown_splitter.split_text(md_text)
    md_header_contents = []
    for split in md_header_splits:
        content = split.page_content
        metadata = split.metadata

        # Check if header exists
        header_exists = bool(metadata)
        if not header_exists:
            continue

        md_header_contents.append({
            "content": content.strip(),
            "length": len(content.strip()),
            "header_level": get_header_level(content),
        })
    return md_header_contents


def merge_header_contents(header_contents: list[dict], max_chars: int = 1000) -> list[dict]:
    merged_header_contents = []
    merged_content = ""
    current_header_level = 0

    header_contents = [header_content["content"]
                       for header_content in header_contents]

    for content in header_contents:
        content_len = len(content)
        header_level = get_header_level(content)

        if len(merged_content) + content_len > max_chars or header_level <= current_header_level:
            if merged_content:
                merged_header_contents.append({
                    "content": merged_content.strip(),
                    "length": len(merged_content.strip()),
                    "header_level": get_header_level(merged_content),
                })
            merged_content = ""
            current_header_level = header_level

        if merged_content:
            merged_content += "\n"  # Add a newline between merged contents
        merged_content += content

    if merged_content:
        merged_header_contents.append({
            "content": merged_content.strip(),
            "length": len(merged_content.strip()),
            "header_level": get_header_level(merged_content),
        })

    return merged_header_contents


def extract_header_contents(md_text: str, max_chars_per_chunk: int = 1000) -> list[dict]:
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
        ("####", "h4"),
        ("#####", "h5"),
        ("######", "h6"),
    ]

    header_contents = get_header_contents(md_text, headers_to_split_on)
    header_contents = merge_header_contents(
        header_contents, max_chars=max_chars_per_chunk)

    # Clean newlines and extra spaces
    for header_content in header_contents:
        header_content["content"] = clean_newlines(header_content["content"])

    return header_contents


def html_to_markdown(
    html_str: str,
    container_selector: str = 'body',
    remove_selectors: list[str] = [],
    replace_selectors: list[dict] = [],
) -> str:
    from bs4 import BeautifulSoup

    # Parse the HTML with BeautifulSoup
    soup = BeautifulSoup(html_str, 'html.parser')

    # Select the container element
    container = soup.select_one(container_selector)

    if not container:
        return ""

    # Remove elements by CSS selector within the container
    for selector in remove_selectors:
        for element in container.select(selector):
            element.decompose()

    # Replace elements by CSS selector within the container
    for replacement in replace_selectors:
        for old_tag, new_tag in replacement.items():
            for element in container.select(old_tag):
                # Create a new tag with the same attributes
                new_element = soup.new_tag(new_tag, **element.attrs)
                # Copy over the contents, not just the string (handles nested tags)
                if element.string is not None:
                    new_element.string = element.string
                else:
                    new_element.extend(element.contents)
                # Replace the old tag with the new tag
                element.replace_with(new_element)

    # Convert the cleaned HTML to Markdown
    markdown = convert_html_to_markdown(str(container))
    markdown = clean_text(markdown)

    # Find the first instance of "# ", then remove all texts before it
    first_header_index = markdown.find("# ")
    if first_header_index != -1:
        markdown = markdown[first_header_index:]

    return markdown


def scrape_markdown(html_str: str) -> dict:
    """Scrape text contents from the HTML string and convert to Markdown, including video URLs under each heading."""

    # Initialize a selector with the input HTML string
    selector = parsel.Selector(text=html_str)

    # Scrape the title from the <title> tag
    page_title = selector.css('title::text').get().strip()

    # Clean and scrape the text content (replace with your actual cleaning logic)
    html_element = selector.css('html')
    # cleaned_html_element = clean_tags(html_element)
    cleaned_html_element = html_element

    # Remove the <title> tag from the cleaned HTML
    cleaned_html_element.css('title').remove()

    # If no <h1> tag exists, add one with the page title
    if not selector.css('h1'):
        cleaned_html_element = parsel.Selector(
            text=f'<h1>{page_title}</h1>{cleaned_html_element.get()}')

    # Extract headings and videos
    headings = []
    all_elements = cleaned_html_element.css('body *')

    current_heading = None
    current_videos = []
    current_content = []  # To store content under each heading

    for element in all_elements:
        element_text = element.css('::text').get() or ""
        # Check if the element is a heading
        if element.root.tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            # Store the previous heading and its content (including videos) if a new heading starts
            if current_heading is not None:
                tag = current_heading['tag']
                level = get_header_level(tag)
                text = clean_text(current_heading['text'])
                content = clean_text('\n'.join(current_content).strip())
                # Prepend n count of '#' to text and content
                text = f"{'#' * level} {text}"
                content = f"{'#' * level} {content}"
                headings.append({
                    'tag': tag,
                    'level': level,
                    'text': text,
                    'video_urls': current_videos,
                    'content': content
                })
            # Start a new heading
            current_heading = {
                'tag': element.root.tag,
                'text': element_text.strip()
            }
            current_videos = []  # Reset videos for new heading
            current_content = []  # Reset content for new heading
        else:
            # Collect the text content under the current heading
            content_text = element_text.strip()
            if content_text and (not current_content or current_content[-1] != content_text):
                current_content.append(content_text)

            # Check if the element contains a YouTube video
            video_url = element.css('iframe::attr(src)').get(
            ) or element.css('a::attr(href)').get()

            # If video URL is valid and not already in the list, add it to both videos and content
            if video_url and "https://www.youtube.com/embed" in video_url and video_url not in current_videos:
                current_videos.append(video_url)

    # Store the last heading and its content (including videos)
    if current_heading is not None:
        tag = current_heading['tag']
        level = get_header_level(tag)
        text = clean_text(current_heading['text'])
        content = clean_text('\n'.join(current_content).strip())
        # Prepend n count of '#' to text and content
        text = f"{'#' * level} {text}"
        content = f"{'#' * level} {content}"
        headings.append({
            'tag': tag,
            'level': level,
            'text': text,
            'video_urls': current_videos,
            'content': content
        })

    # Get the cleaned HTML as a string
    cleaned_html_str = cleaned_html_element.get()

    # Convert the cleaned HTML to markdown
    markdown_text = convert_html_to_markdown(cleaned_html_str)
    markdown_text = clean_text(markdown_text)

    # Combine all heading contents into one string for comparison
    # combined_content = '\n'.join([heading['content']
    #                              for heading in headings]).strip()

    # Remove duplicate video URLs except for the last occurrence
    video_url_headings_dict = {}
    for heading in headings:
        for video_url in heading['video_urls']:
            if video_url in video_url_headings_dict:
                video_url_headings_dict[video_url].append(heading)
            else:
                video_url_headings_dict[video_url] = [heading]

    for video_url, video_url_headings in video_url_headings_dict.items():
        for heading in video_url_headings[:-1]:
            heading['video_urls'].remove(video_url)

        last_heading = video_url_headings[-1]
        next_heading_index = headings.index(last_heading) + 1
        next_heading = headings[next_heading_index] if next_heading_index < len(
            headings) else None
        # Update markdown text with video URLs at the end of each last_heading content
        # Find the index of the last_heading text in the markdown text and the index of the next last_heading text
        heading_text_next_index = markdown_text.find(
            next_heading['text']) if next_heading else len(markdown_text)
        # Update heading_text_next_index to deduct 1 for the newline character
        heading_text_next_index -= 1

        # Add the video URL before the next heading text
        video_url_markdown = f"\n[Reference video]({video_url})"
        markdown_text = markdown_text[:heading_text_next_index] + \
            video_url_markdown + markdown_text[heading_text_next_index:]

    # Remove all empty texts
    headings = [heading for heading in headings if heading['text'].strip()[
        heading['level']:]]
    # Remove all text before the first heading level 1
    if headings:
        # Find heading with level 1
        first_heading_index = next(
            (i for i, heading in enumerate(headings) if heading['level'] == 1), None)
        first_heading = headings[first_heading_index]
        # Find the index of the first heading text in the markdown text
        heading_text_index = markdown_text.find(first_heading['text'])
        # Update heading_text_index to deduct 1 for the newline character
        heading_text_index -= 1 if heading_text_index > 0 else 0
        # Remove all text before the first heading
        markdown_text = markdown_text[heading_text_index:].strip()

    return {
        "title": page_title,
        "content": markdown_text,
        "headings": headings,
    }


__all__ = [
    "convert_html_to_markdown",
    # "get_header_level",
    # "get_header_contents",
    # "merge_header_contents",
    # "extract_header_contents",
    "html_to_markdown",
    "scrape_markdown",
]

if __name__ == '__main__':
    valid_id = "passport"
    markdown_file = f"generated/{valid_id}/_main.md"
    output_preprocessed_file = "generated/_samples/_main_preprocessed.json"
    output_info_file = "generated/_samples/_main_info.json"

    with open(markdown_file, "r") as f:
        md_text = f.read()

    max_chars_per_chunk = 2048
    header_contents = extract_header_contents(md_text, max_chars_per_chunk)

    min_content_length = min(
        [header_content["length"] for header_content in header_contents])
    max_content_length = max(
        [header_content["length"] for header_content in header_contents])
    average_content_length = math.ceil(sum(
        [header_content["length"] for header_content in header_contents]) / len(header_contents))
    total_content_length = sum(
        [header_content["length"] for header_content in header_contents])
    info_dict = {
        "valid_id": valid_id,
        "chunks": len(header_contents),
        "lengths": {
            "min": min_content_length,
            "max": max_content_length,
            "average": average_content_length,
            "total": total_content_length,
        },
        "contents": header_contents,
    }

    with open(output_info_file, "w") as f:
        json.dump(info_dict, f, indent=2, ensure_ascii=False)
    logger.success(f"Info saved to: {output_info_file}")

    preprocessed_contents: list[str] = []
    for header_content in header_contents:
        preprocessed_contents.append(header_content["content"])
    with open(output_preprocessed_file, "w") as f:
        json.dump(preprocessed_contents, f, indent=2, ensure_ascii=False)
    logger.success(f"Preprocessed contents saved to: {
                   output_preprocessed_file}")
