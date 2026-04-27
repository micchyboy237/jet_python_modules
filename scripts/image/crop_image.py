import argparse
import os

from PIL import Image, ImageDraw


def crop_and_save(input_path, output_dir):
    img = Image.open(input_path)
    w, h = img.size
    side = min(w, h)

    # Center crop square
    left = (w - side) // 2
    top = (h - side) // 2
    square = img.crop((left, top, left + side, top + side))

    os.makedirs(output_dir, exist_ok=True)
    square_path = os.path.join(output_dir, "profile_square.png")
    square.save(square_path)

    # Create circular version
    mask = Image.new("L", (side, side), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, side, side), fill=255)

    circle = Image.new("RGBA", (side, side))
    circle.paste(square, (0, 0), mask=mask)

    circle_path = os.path.join(output_dir, "profile_circle.png")
    circle.save(circle_path)

    print("Saved files:")
    print("Square: ", square_path)
    print("Circle: ", circle_path)
    return square_path, circle_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crop image to square and circle for profile."
    )
    parser.add_argument("input_image", help="Path to the input image file")
    parser.add_argument(
        "--output_dir",
        default="cropped_square_circle",
        help="Output directory to save the images (default: cropped_square_circle)",
    )
    args = parser.parse_args()
    crop_and_save(args.input_image, args.output_dir)
