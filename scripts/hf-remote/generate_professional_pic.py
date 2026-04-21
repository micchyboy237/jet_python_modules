import io
import os

import requests
from PIL import Image

API_URL = "https://api-inference.huggingface.co/models/timbrooks/instruct-pix2pix"
HEADERS = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}


def edit_image(input_path: str, output_path: str):
    with open(input_path, "rb") as f:
        image_bytes = f.read()

    payload = {
        "inputs": "Add a professional black suit and white shirt, corporate headshot, studio lighting"
    }

    response = requests.post(API_URL, headers=HEADERS, data=image_bytes, params=payload)

    if response.status_code != 200:
        raise Exception(response.text)

    image = Image.open(io.BytesIO(response.content))
    image.save(output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a professional-looking photo by editing a given input image."
    )
    parser.add_argument("input_path", type=str, help="Path to the input image file")
    parser.add_argument(
        "output_path",
        type=str,
        nargs="?",
        default="generated_professional_look.png",
        help="Path to save the output image",
    )
    args = parser.parse_args()

    edit_image(args.input_path, args.output_path)
