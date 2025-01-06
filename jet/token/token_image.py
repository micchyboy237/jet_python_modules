import base64
import io
import struct
from typing import Literal, Union
from jet.token.http_handler import HTTPHandler


def resize_image_high_res(width, height):
    # Maximum dimensions for high res mode
    max_short_side = 768
    max_long_side = 2000

    # Return early if no resizing is needed
    if width <= 768 and height <= 768:
        return width, height

    # Determine the longer and shorter sides
    longer_side = max(width, height)
    shorter_side = min(width, height)

    # Calculate the aspect ratio
    aspect_ratio = longer_side / shorter_side

    # Resize based on the short side being 768px
    if width <= height:  # Portrait or square
        resized_width = max_short_side
        resized_height = int(resized_width * aspect_ratio)
        # if the long side exceeds the limit after resizing, adjust both sides accordingly
        if resized_height > max_long_side:
            resized_height = max_long_side
            resized_width = int(resized_height / aspect_ratio)
    else:  # Landscape
        resized_height = max_short_side
        resized_width = int(resized_height * aspect_ratio)
        # if the long side exceeds the limit after resizing, adjust both sides accordingly
        if resized_width > max_long_side:
            resized_width = max_long_side
            resized_height = int(resized_width / aspect_ratio)

    return resized_width, resized_height


def calculate_tiles_needed(
    resized_width, resized_height, tile_width=512, tile_height=512
):
    tiles_across = (resized_width + tile_width - 1) // tile_width
    tiles_down = (resized_height + tile_height - 1) // tile_height
    total_tiles = tiles_across * tiles_down
    return total_tiles


def get_image_type(image_data: bytes) -> Union[str, None]:
    """take an image (really only the first ~100 bytes max are needed)
    and return 'png' 'gif' 'jpeg' 'heic' or None. method added to
    allow deprecation of imghdr in 3.13"""

    if image_data[0:8] == b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a":
        return "png"

    if image_data[0:4] == b"GIF8" and image_data[5:6] == b"a":
        return "gif"

    if image_data[0:3] == b"\xff\xd8\xff":
        return "jpeg"

    if image_data[4:8] == b"ftyp":
        return "heic"

    return None


def get_image_dimensions(data):
    img_data = None

    try:
        # Try to open as URL
        # Try to open as URL
        client = HTTPHandler(concurrent_limit=1)
        response = client.get(data)
        img_data = response.read()
    except Exception:
        # If not URL, assume it's base64
        header, encoded = data.split(",", 1)
        img_data = base64.b64decode(encoded)

    img_type = get_image_type(img_data)

    if img_type == "png":
        w, h = struct.unpack(">LL", img_data[16:24])
        return w, h
    elif img_type == "gif":
        w, h = struct.unpack("<HH", img_data[6:10])
        return w, h
    elif img_type == "jpeg":
        with io.BytesIO(img_data) as fhandle:
            fhandle.seek(0)
            size = 2
            ftype = 0
            while not 0xC0 <= ftype <= 0xCF or ftype in (0xC4, 0xC8, 0xCC):
                fhandle.seek(size, 1)
                byte = fhandle.read(1)
                while ord(byte) == 0xFF:
                    byte = fhandle.read(1)
                ftype = ord(byte)
                size = struct.unpack(">H", fhandle.read(2))[0] - 2
            fhandle.seek(1, 1)
            h, w = struct.unpack(">HH", fhandle.read(4))
        return w, h
    else:
        return None, None


def calculate_img_tokens(
    data,
    mode: Literal["low", "high", "auto"] = "auto",
    base_tokens: int = 85,  # openai default - https://openai.com/pricing
):
    if mode == "low" or mode == "auto":
        return base_tokens
    elif mode == "high":
        width, height = get_image_dimensions(data=data)
        resized_width, resized_height = resize_image_high_res(
            width=width, height=height
        )
        tiles_needed_high_res = calculate_tiles_needed(
            resized_width, resized_height)
        tile_tokens = (base_tokens * 2) * tiles_needed_high_res
        total_tokens = base_tokens + tile_tokens
        return total_tokens
