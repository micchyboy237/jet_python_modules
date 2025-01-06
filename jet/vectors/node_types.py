from typing import TypedDict, Optional, Union


class TextNodeAttributes(TypedDict, total=False):
    text: str
    text_length: int
    start_end: tuple[int, int]


class ImageNodeAttributes(TypedDict, total=False):
    image: str
    image_path: str
    image_url: str
    image_mimetype: str


class SourceNodeAttributes(TypedDict):
    node_id: str
    metadata: Optional[dict]
    score: Optional[float]
    text: Optional[str]
    text_length: Optional[int]
    start_end: Optional[tuple[int, int]]
    image_info: Optional[ImageNodeAttributes]
