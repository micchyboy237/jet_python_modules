# generate_professional_pic.py
import base64
import io
import os
import time
from pathlib import Path
from typing import Tuple

import requests
from PIL import Image
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tqdm import tqdm

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem

REPLICATE_API_URL = "https://api.replicate.com/v1/predictions"
REPLICATE_VERSION = "30c1d0b916a6f8efce20493f5d61ee27491ab2a60437c13c588468b9810ec23f"  # stable version of instruct-pix2pix

PROMPT = "Add a tailored black suit and colored shirt with tie, corporate headshot, studio lighting"

console = Console()


# -------------------------
# Utilities (unchanged)
# -------------------------
def get_image_info(image_bytes: bytes) -> Tuple[int, Tuple[int, int]]:
    img = Image.open(io.BytesIO(image_bytes))
    size_kb = len(image_bytes) // 1024
    return size_kb, img.size


def log_image_comparison(original_bytes: bytes, processed_bytes: bytes):
    orig_size, orig_res = get_image_info(original_bytes)
    proc_size, proc_res = get_image_info(processed_bytes)
    table = Table(title="Image Optimization Summary")
    table.add_column("Type")
    table.add_column("Size (KB)")
    table.add_column("Resolution")
    table.add_row("Original", str(orig_size), f"{orig_res[0]}x{orig_res[1]}")
    table.add_row("Processed", str(proc_size), f"{proc_res[0]}x{proc_res[1]}")
    console.print(table)


def preprocess_image(
    path: str, max_size: Tuple[int, int] = (768, 768), quality: int = 85
) -> Tuple[bytes, bytes]:
    with open(path, "rb") as f:
        original_bytes = f.read()
    img = Image.open(io.BytesIO(original_bytes)).convert("RGB")
    img.thumbnail(max_size)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality, optimize=True)
    return original_bytes, buffer.getvalue()


def save_image(image_bytes: bytes, path: str):
    img = Image.open(io.BytesIO(image_bytes))
    img.save(path)


def encode_image_bytes(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


# -------------------------
# New Replicate call (async + polling)
# -------------------------
def call_model(image_base64: str) -> bytes:
    headers = {
        "Authorization": f"Token {os.getenv('REPLICATE_API_KEY')}",
        "Content-Type": "application/json",
    }

    payload = {
        "version": REPLICATE_VERSION,
        "input": {
            "image": f"data:image/jpeg;base64,{image_base64}",
            "prompt": PROMPT,
            "num_inference_steps": 20,
            "image_guidance_scale": 1.5,
            "guidance_scale": 7.5,
        },
    }

    console.log("[bold blue]Sending request to Replicate...[/bold blue]")
    response = requests.post(REPLICATE_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    prediction = response.json()

    prediction_id = prediction["id"]
    console.log(f"[cyan]Prediction started → ID: {prediction_id}[/cyan]")

    # Poll until done
    while True:
        time.sleep(2)
        resp = requests.get(f"{REPLICATE_API_URL}/{prediction_id}", headers=headers)
        resp.raise_for_status()
        data = resp.json()
        status = data["status"]

        if status == "succeeded":
            console.log("[green]✅ Generation completed[/green]")
            image_url = data["output"][0]
            break
        elif status == "failed":
            raise Exception(f"Replicate failed: {data.get('error')}")
        else:
            console.log(f"[yellow]Waiting... status: {status}[/yellow]")

    # Download final image
    img_resp = requests.get(image_url)
    img_resp.raise_for_status()
    return img_resp.content


# -------------------------
# Pipeline (same structure as before)
# -------------------------
def edit_image(input_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    console.print(
        Panel.fit(
            f"[bold]Input:[/bold] {input_path}\n[bold]Output:[/bold] {output_dir}",
            title="Run Configuration",
            style="blue",
        )
    )

    steps = [
        "Preprocessing image",
        "Saving compressed image",
        "Encoding image",
        "Calling Replicate",
        "Saving output",
    ]

    with tqdm(total=len(steps), desc="Processing", unit="step") as pbar:
        original_bytes, processed_bytes = preprocess_image(input_path)
        console.log("[cyan]Image preprocessed[/cyan]")
        log_image_comparison(original_bytes, processed_bytes)
        pbar.update(1)

        compressed_path = os.path.join(output_dir, "compressed.png")
        save_image(processed_bytes, compressed_path)
        console.log(f"[green]Compressed image saved:[/green] {compressed_path}")
        pbar.update(1)

        image_base64 = encode_image_bytes(processed_bytes)
        console.log(f"[cyan]Encoded image size:[/cyan] {len(image_base64) // 1024} KB")
        pbar.update(1)

        output_bytes = call_model(image_base64)
        pbar.update(1)

        output_path = os.path.join(output_dir, "output.png")
        save_image(output_bytes, output_path)
        console.log(f"[bold green]Output saved:[/bold green] {output_path}")
        pbar.update(1)

    console.print(Panel.fit("✔ Process completed successfully", style="green"))


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    import argparse
    import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("--output_dir", "-o", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = OUTPUT_DIR / f"run_{timestamp}"
    else:
        output_dir = Path(args.output_dir)
    output_dir = str(output_dir)

    edit_image(args.input_path, output_dir)
