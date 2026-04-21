# generate_professional_pic.py

import base64
import io
import os

import requests
from PIL import Image

API_URL = "https://api-inference.huggingface.co/models/timbrooks/instruct-pix2pix"
HEADERS = {
    "Authorization": f"Bearer {os.getenv('HF_TOKEN')}",
    "Content-Type": "application/json",
}


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def edit_image(input_path: str, output_path: str):
    image_base64 = encode_image(input_path)

    payload = {
        "inputs": {
            "image": image_base64,
            "prompt": "A young professional man in a tailored black suit and white shirt, confident corporate headshot, modern studio lighting, sharp details, photorealistic, 8k",
        }
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload)

    if response.status_code != 200:
        raise Exception(response.text)

    image = Image.open(io.BytesIO(response.content))
    image.save(output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("output_path", type=str, nargs="?", default="output.png")
    args = parser.parse_args()

    edit_image(args.input_path, args.output_path)
