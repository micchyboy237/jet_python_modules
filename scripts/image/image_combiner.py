"""
Image Combiner for Bypassing AI Chat Upload Limits
===================================================
Combines multiple images into a single file (PDF or Image) to bypass
upload attachment limits in AI chat interfaces like Qwen.ai.

Requirements:
    pip install Pillow img2pdf

Usage Examples:
    # Minimal usage - auto-detects mode and output
    python image_combiner.py --input ./screenshots

    # Create a multi-page PDF
    python image_combiner.py --input ./screenshots --output combined.pdf

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
from typing import List, Optional, Tuple

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


def load_images(
    file_paths: List[str], preserve_alpha: bool = False
) -> List[Image.Image]:
    """
    Load and normalize images from file paths.
    Optionally preserves alpha channels for PNG/WebP output.

    Args:
        file_paths: List of image file paths
        preserve_alpha: If True, keep RGBA/LA modes

    Returns:
        List of loaded PIL Image objects
    """
    images = []
    for path in file_paths:
        try:
            img = Image.open(path)
            original_mode = img.mode

            # Convert palette mode always
            if img.mode == "P":
                img = img.convert("RGBA" if preserve_alpha else "RGB")
                logger.debug(
                    f"Converted {os.path.basename(path)} from P to {'RGBA' if preserve_alpha else 'RGB'}"
                )
            # Convert LA to RGBA if preserving alpha
            elif img.mode == "LA" and preserve_alpha:
                img = img.convert("RGBA")
                logger.debug(f"Converted {os.path.basename(path)} from LA to RGBA")
            # Convert to RGB only if not preserving alpha
            elif not preserve_alpha and img.mode in ("RGBA", "LA"):
                img = img.convert("RGB")
                logger.debug(
                    f"Converted {os.path.basename(path)} from {original_mode} to RGB"
                )

            images.append(img)
            logger.info(
                f"  ✓ Loaded: {os.path.basename(path)} ({img.size[0]}x{img.size[1]}, {img.mode})"
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
    bg_color: Optional[Tuple[int, int, int]] = (255, 255, 255),
    max_width: Optional[int] = None,
) -> Image.Image:
    """Merge images vertically with optional spacing."""
    # Resize if max_width specified
    if max_width:
        resized = []
        for img in images:
            if img.width > max_width:
                ratio = max_width / img.width
                new_size = (max_width, int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            resized.append(img)
        images = resized

    max_width_actual = max(img.width for img in images)
    total_height = sum(img.height for img in images) + spacing * (len(images) - 1)

    # Determine mode based on first image and bg_color
    has_alpha = any(img.mode in ("RGBA", "LA") for img in images)
    mode = "RGBA" if (has_alpha and bg_color is None) else "RGB"

    result = Image.new(
        mode, (max_width_actual, total_height), bg_color if bg_color else (0, 0, 0, 0)
    )

    y_offset = 0
    for img in images:
        # Ensure compatible mode for pasting
        if mode == "RGBA" and img.mode != "RGBA":
            img = img.convert("RGBA")
        elif mode == "RGB" and img.mode == "RGBA":
            img = img.convert("RGB")

        x_offset = (max_width_actual - img.width) // 2  # Center horizontally
        result.paste(img, (x_offset, y_offset))
        y_offset += img.height + spacing

    logger.info(f"Vertical merge complete: {result.size[0]}x{result.size[1]}")
    return result


def merge_horizontal(
    images: List[Image.Image],
    spacing: int = 0,
    bg_color: Optional[Tuple[int, int, int]] = (255, 255, 255),
    max_height: Optional[int] = None,
) -> Image.Image:
    """Merge images horizontally with optional spacing."""
    # Resize if max_height specified
    if max_height:
        resized = []
        for img in images:
            if img.height > max_height:
                ratio = max_height / img.height
                new_size = (int(img.width * ratio), max_height)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            resized.append(img)
        images = resized

    max_height_actual = max(img.height for img in images)
    total_width = sum(img.width for img in images) + spacing * (len(images) - 1)

    # Determine mode based on first image and bg_color
    has_alpha = any(img.mode in ("RGBA", "LA") for img in images)
    mode = "RGBA" if (has_alpha and bg_color is None) else "RGB"

    result = Image.new(
        mode, (total_width, max_height_actual), bg_color if bg_color else (0, 0, 0, 0)
    )

    x_offset = 0
    for img in images:
        # Ensure compatible mode for pasting
        if mode == "RGBA" and img.mode != "RGBA":
            img = img.convert("RGBA")
        elif mode == "RGB" and img.mode == "RGBA":
            img = img.convert("RGB")

        y_offset = (max_height_actual - img.height) // 2  # Center vertically
        result.paste(img, (x_offset, y_offset))
        x_offset += img.width + spacing

    logger.info(f"Horizontal merge complete: {result.size[0]}x{result.size[1]}")
    return result


def merge_grid(
    images: List[Image.Image],
    cols: int = 3,
    spacing: int = 10,
    bg_color: Optional[Tuple[int, int, int]] = (255, 255, 255),
    cell_width: Optional[int] = None,
    cell_height: Optional[int] = None,
) -> Image.Image:
    """Arrange images in a uniform grid layout."""
    rows = (len(images) + cols - 1) // cols

    # Use provided cell size or calculate from largest image
    if cell_width is None or cell_height is None:
        cell_width = max(img.width for img in images)
        cell_height = max(img.height for img in images)

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

    # Determine mode
    has_alpha = any(img.mode in ("RGBA", "LA") for img in images)
    mode = "RGBA" if (has_alpha and bg_color is None) else "RGB"

    result = Image.new(mode, (total_w, total_h), bg_color if bg_color else (0, 0, 0, 0))

    for idx, img in enumerate(resized):
        row, col = divmod(idx, cols)
        x = col * (cell_width + spacing) + (cell_width - img.width) // 2
        y = row * (cell_height + spacing) + (cell_height - img.height) // 2

        # Ensure compatible mode
        if mode == "RGBA" and img.mode != "RGBA":
            img = img.convert("RGBA")
        elif mode == "RGB" and img.mode == "RGBA":
            img = img.convert("RGB")

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
        # JPEG doesn't support alpha
        if image.mode == "RGBA":
            image = image.convert("RGB")
            logger.info("Converted RGBA to RGB for JPEG compatibility")
        save_kwargs["quality"] = quality
        save_kwargs["optimize"] = True
    elif ext == ".png":
        save_kwargs["optimize"] = True
    elif ext == ".webp":
        save_kwargs["quality"] = quality

    image.save(output_path, **save_kwargs)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"Image saved: {output_path} ({size_mb:.2f} MB)")


def infer_mode_from_output(output_path: str) -> str:
    """Infer merge mode from output file extension."""
    ext = Path(output_path).suffix.lower()
    if ext == ".pdf":
        return "pdf"
    return "vertical"  # Default for image formats


def generate_default_output(input_path: str, mode: str = "vertical") -> str:
    """Generate default output filename based on input."""
    input_name = Path(input_path).stem

    if os.path.isfile(input_path):
        # Single file input
        base_name = input_name
    else:
        # Directory input
        base_name = Path(input_path).name

    if mode == "pdf":
        return f"{base_name}_combined.pdf"
    else:
        return f"{base_name}_merged.png"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine multiple images to bypass AI chat upload limits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["vertical", "horizontal", "grid", "pdf"],
        help="Combination mode (auto-detected from output if not specified)",
    )
    parser.add_argument(
        "--input", required=True, help="Input directory or single image file"
    )
    parser.add_argument(
        "--output", help="Output file path (auto-generated if not specified)"
    )
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
    parser.add_argument(
        "--max-width",
        type=int,
        default=None,
        help="Maximum width for output image (prevents oversized files)",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=None,
        help="Maximum height for output image (prevents oversized files)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("Image Combiner - Bypass AI Upload Limits")
    logger.info(f"Input: {args.input}")
    logger.info("=" * 60)

    # Step 1: Discover images
    try:
        image_files = get_image_files(args.input)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)

    # Step 2: Determine mode and output
    if args.output:
        output_path = args.output
        # Auto-detect mode from output extension if not specified
        if not args.mode:
            args.mode = infer_mode_from_output(output_path)
            logger.info(
                f"Auto-detected mode: {args.mode} (from .{Path(output_path).suffix})"
            )
    else:
        # Generate default output
        args.mode = args.mode or "vertical"
        output_path = generate_default_output(args.input, args.mode)
        logger.info(f"Auto-generated output: {output_path}")
        logger.info(f"Using mode: {args.mode}")

    # Ensure PDF extension matches mode
    if args.mode == "pdf" and not output_path.lower().endswith(".pdf"):
        output_path = str(Path(output_path).with_suffix(".pdf"))
        logger.info(f"Adjusted output to PDF: {output_path}")

    logger.info(f"Mode: {args.mode} | Output: {output_path}")

    # Step 3: Process based on mode
    if args.mode == "pdf":
        logger.info("Creating PDF (lossless embedding)...")
        create_pdf(image_files, output_path)
    else:
        logger.info(f"Loading {len(image_files)} images...")

        # Preserve alpha for PNG/WebP outputs
        output_ext = Path(output_path).suffix.lower()
        preserve_alpha = (
            output_ext in (".png", ".webp") and args.bg_color == "transparent"
        )

        images = load_images(image_files, preserve_alpha=preserve_alpha)

        bg_color = BG_COLORS[args.bg_color]
        logger.info(
            f"Merging in '{args.mode}' mode (spacing={args.spacing}px, bg={args.bg_color})..."
        )

        merge_funcs = {
            "vertical": lambda: merge_vertical(
                images, args.spacing, bg_color, max_width=args.max_width
            ),
            "horizontal": lambda: merge_horizontal(
                images, args.spacing, bg_color, max_height=args.max_height
            ),
            "grid": lambda: merge_grid(images, args.cols, args.spacing, bg_color),
        }

        result = merge_funcs[args.mode]()
        save_image(result, output_path, args.quality)

    logger.info("=" * 60)
    logger.info("✓ Done! Upload this single file to your AI chat.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
