from bs4 import BeautifulSoup


def clean_html_by_tags(html_string: str, exclude_tags: list) -> str:
    """
    Remove specified HTML tags from an HTML string while keeping their content.

    Args:
        html_string (str): The input HTML string
        exclude_tags (list): List of tag names to remove (e.g., ['script', 'style', 'div'])

    Returns:
        str: Cleaned HTML string with specified tags removed
    """
    # Parse the HTML
    soup = BeautifulSoup(html_string, "html.parser")

    # Find and remove all specified tags
    for tag_name in exclude_tags:
        # Find all elements with the specified tag name
        for element in soup.find_all(tag_name):
            # Use decompose() to completely remove the tag and its contents
            element.decompose()

    # Return the cleaned HTML as a string
    return str(soup)
