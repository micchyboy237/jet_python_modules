"""
Image Combiner for Bypassing AI Chat Upload Limits
===================================================
Combines multiple images into a single file (PDF or Image) to bypass
upload attachment limits in AI chat interfaces like Qwen.ai.

Requirements:
    pip install Pillow img2pdf

Usage Examples:
    # Create a multi-page PDF (Best for AI analysis)
    python image_combiner.py --mode pdf --input ./screenshots --output combined.pdf

    # Vertical stack with spacing
    python image_combiner.py --mode vertical --input ./images --output merged.png --spacing 10

    # 3-column grid collage
    python image_combiner.py --mode grid --input ./images --cols 3 --output grid.png

    # Horizontal merge with black background
    python image_combiner.py --mode horizontal --input ./images --output side_by_side.jpg --bg-color black
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

try:
    from PIL import Image
except ImportError:
    logger.error("Pillow not installed. Run: pip install Pillow")
    sys.exit(1)

try:
    import img2pdf
except ImportError:
    logger.error("img2pdf not installed. Run: pip install img2pdf")
    sys.exit(1)


# Supported image extensions
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

# Background color presets
BG_COLORS = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "gray": (128, 128, 128),
    "transparent": None,  # Only for PNG output
}


def get_image_files(input_path: str) -> List[str]:
    """
    Get sorted list of image files from a directory.

    Args:
        input_path: Directory path containing images

    Returns:
        Sorted list of absolute image file paths

    Raises:
        FileNotFoundError: If no images found
    """
    if os.path.isfile(input_path):
        ext = Path(input_path).suffix.lower()
        if ext in SUPPORTED_EXTENSIONS:
            logger.info(f"Single file mode: {input_path}")
            return [os.path.abspath(input_path)]
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    image_files = []
    for entry in os.scandir(input_path):
        if entry.is_file() and Path(entry.name).suffix.lower() in SUPPORTED_EXTENSIONS:
            image_files.append(entry.path)

    if not image_files:
        raise FileNotFoundError(f"No supported image files found in: {input_path}")

    # Natural sort by filename
    image_files.sort(key=lambda x: os.path.basename(x).lower())
    logger.info(f"Found {len(image_files)} images in {input_path}")
    return image_files


def load_images(file_paths: List[str]) -> List[Image.Image]:
    """
    Load and normalize images from file paths.
    Converts RGBA/P modes to RGB for JPEG compatibility.

    Args:
        file_paths: List of image file paths

    Returns:
        List of loaded PIL Image objects
    """
    images = []
    for path in file_paths:
        try:
            img = Image.open(path)
            original_mode = img.mode

            # Convert to RGB for consistent processing
            if img.mode in ("RGBA", "P", "LA"):
                img = img.convert("RGB")
                logger.debug(
                    f"Converted {os.path.basename(path)} from {original_mode} to RGB"
                )

            images.append(img)
            logger.info(
                f"  ✓ Loaded: {os.path.basename(path)} ({img.size[0]}x{img.size[1]}, {original_mode})"
            )
        except Exception as e:
            logger.warning(f"  ✗ Skipped {os.path.basename(path)}: {e}")

    if not images:
        raise ValueError("No valid images could be loaded")

    logger.info(f"Successfully loaded {len(images)}/{len(file_paths)} images")
    return images


def merge_vertical(
    images: List[Image.Image],
    spacing: int = 0,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Merge images vertically with optional spacing."""
    max_width = max(img.width for img in images)
    total_height = sum(img.height for img in images) + spacing * (len(images) - 1)

    result = Image.new("RGB", (max_width, total_height), bg_color)

    y_offset = 0
    for img in images:
        x_offset = (max_width - img.width) // 2  # Center horizontally
        result.paste(img, (x_offset, y_offset))
        y_offset += img.height + spacing

    logger.info(f"Vertical merge complete: {result.size[0]}x{result.size[1]}")
    return result


def merge_horizontal(
    images: List[Image.Image],
    spacing: int = 0,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Merge images horizontally with optional spacing."""
    max_height = max(img.height for img in images)
    total_width = sum(img.width for img in images) + spacing * (len(images) - 1)

    result = Image.new("RGB", (total_width, max_height), bg_color)

    x_offset = 0
    for img in images:
        y_offset = (max_height - img.height) // 2  # Center vertically
        result.paste(img, (x_offset, y_offset))
        x_offset += img.width + spacing

    logger.info(f"Horizontal merge complete: {result.size[0]}x{result.size[1]}")
    return result


def merge_grid(
    images: List[Image.Image],
    cols: int = 3,
    spacing: int = 10,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Arrange images in a uniform grid layout."""
    rows = (len(images) + cols - 1) // cols

    # Use first image as reference cell size
    cell_width = images[0].width
    cell_height = images[0].height

    # Resize all images to fit cell while maintaining aspect ratio
    resized = []
    for img in images:
        img_ratio = img.width / img.height
        cell_ratio = cell_width / cell_height

        if img_ratio > cell_ratio:
            new_w = cell_width
            new_h = int(cell_width / img_ratio)
        else:
            new_h = cell_height
            new_w = int(cell_height * img_ratio)

        r = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        resized.append(r)

    # Calculate total dimensions
    total_w = cols * cell_width + (cols - 1) * spacing
    total_h = rows * cell_height + (rows - 1) * spacing

    result = Image.new("RGB", (total_w, total_h), bg_color)

    for idx, img in enumerate(resized):
        row, col = divmod(idx, cols)
        x = col * (cell_width + spacing) + (cell_width - img.width) // 2
        y = row * (cell_height + spacing) + (cell_height - img.height) // 2
        result.paste(img, (x, y))

    logger.info(
        f"Grid merge complete: {cols}x{rows} grid, {result.size[0]}x{result.size[1]}"
    )
    return result


def create_pdf(image_files: List[str], output_path: str) -> None:
    """
    Create PDF from images using img2pdf (lossless embedding).
    Each image becomes a separate page.
    """
    with open(output_path, "wb") as f:
        f.write(img2pdf.convert(image_files))

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(
        f"PDF created: {output_path} ({size_mb:.2f} MB, {len(image_files)} pages)"
    )


def save_image(image: Image.Image, output_path: str, quality: int = 95) -> None:
    """Save merged image with appropriate format handling."""
    ext = Path(output_path).suffix.lower()

    save_kwargs = {}
    if ext in (".jpg", ".jpeg"):
        save_kwargs["quality"] = quality
        save_kwargs["optimize"] = True
    elif ext == ".png":
        save_kwargs["optimize"] = True
    elif ext == ".webp":
        save_kwargs["quality"] = quality

    image.save(output_path, **save_kwargs)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"Image saved: {output_path} ({size_mb:.2f} MB)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine multiple images to bypass AI chat upload limits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["vertical", "horizontal", "grid", "pdf"],
        help="Combination mode",
    )
    parser.add_argument(
        "--input", required=True, help="Input directory or single image file"
    )
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument(
        "--cols", type=int, default=3, help="Columns for grid mode (default: 3)"
    )
    parser.add_argument(
        "--spacing",
        type=int,
        default=10,
        help="Spacing between images in px (default: 10)",
    )
    parser.add_argument(
        "--bg-color",
        default="white",
        choices=list(BG_COLORS.keys()),
        help="Background color (default: white)",
    )
    parser.add_argument(
        "--quality", type=int, default=95, help="JPEG/WebP quality 1-100 (default: 95)"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("Image Combiner - Bypass AI Upload Limits")
    logger.info(f"Mode: {args.mode} | Input: {args.input}")
    logger.info("=" * 60)

    # Step 1: Discover images
    try:
        image_files = get_image_files(args.input)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)

    # Step 2: Process based on mode
    if args.mode == "pdf":
        logger.info("Creating PDF (lossless embedding)...")
        create_pdf(image_files, args.output)
    else:
        logger.info(f"Loading {len(image_files)} images...")
        images = load_images(image_files)

        bg_color = BG_COLORS[args.bg_color]
        logger.info(
            f"Merging in '{args.mode}' mode (spacing={args.spacing}px, bg={args.bg - color})..."
        )

        merge_funcs = {
            "vertical": lambda: merge_vertical(images, args.spacing, bg_color),
            "horizontal": lambda: merge_horizontal(images, args.spacing, bg_color),
            "grid": lambda: merge_grid(images, args.cols, args.spacing, bg_color),
        }

        result = merge_funcs[args.mode]()
        save_image(result, args.output, args.quality)

    logger.info("=" * 60)
    logger.info("✓ Done! Upload this single file to your AI chat.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
