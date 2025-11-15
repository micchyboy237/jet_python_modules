from typing import Literal


def display_iterm2_image(
    png_data: bytes,
    *,
    width: int | Literal["auto"] = "auto",
    height: int | Literal["auto"] = 150,
    preserve_aspect_ratio: bool = True,
) -> None:
    """
    Display PNG image inline in iTerm2 using imgcat protocol.

    Args:
        png_data: Raw PNG bytes.
        width: Target width in pixels or 'auto' (default: 'auto').
        height: Target height in pixels or 'auto' (default: 150).
        preserve_aspect_ratio: Keep aspect ratio when one dimension is specified (default: True).
    """
    import base64

    b64 = base64.b64encode(png_data).decode()

    params = ["inline=1"]
    if width != "auto":
        params.append(f"width={width}")
    if height != "auto":
        params.append(f"height={height}")
    if width != "auto" or height != "auto":
        params.append(f"preserveAspectRatio={'1' if preserve_aspect_ratio else '0'}")

    print(f"\033]1337;File={';'.join(params)}:{b64}\a", end="")
    print()
