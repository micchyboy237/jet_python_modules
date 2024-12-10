import re


def to_snake_case(url: str) -> str:
    url = url.replace("https://", "").replace("http://", "")

    url_segments = url.split("/")

    # Convert to lower case
    # Remove non-alphanumeric characters
    # Strip underscore from edges
    transformed_segments = []
    for segment in url_segments:
        segment = segment.lower()
        segment = snake_case_path = re.sub(r'[^a-zA-Z0-9/]', '_', segment)
        segment = snake_case_path.strip("_")
        transformed_segments.append(segment)

    joined_segments = "_".join(transformed_segments)
    joined_segments = joined_segments.strip("_")
    # Replace consecutive underscores with a single underscore
    joined_segments = re.sub(r'_{2,}', '_', joined_segments)
    return joined_segments
