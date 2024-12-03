import re
from urllib.parse import urlparse


def to_snake_case(url: str) -> str:
    # Parse the URL to get the path
    parsed_url = urlparse(url)
    path = parsed_url.path

    # Replace slashes with underscores, convert to lowercase, and remove non-alphanumeric characters
    snake_case_url = re.sub(r'[^a-zA-Z0-9/]', '_', path)
    snake_case_url = snake_case_url.strip('_').lower().replace('/', '_')

    return snake_case_url
