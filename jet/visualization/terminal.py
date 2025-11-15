from typing import Literal

SizeUnit = Literal["auto"] | int | float | str  # int/float -> treated as cells (e.g. 20), str must include unit like "200px", "50%", "20"

def display_iterm2_image(
    png_data: bytes,
    *,
    width: SizeUnit = "auto",
    height: SizeUnit = "50%",
    preserve_aspect_ratio: bool = True,
) -> None:
    """
    Display PNG image inline in iTerm2 using imgcat protocol.

    Args:
        png_data: Raw PNG bytes.
        width: Target width - 'auto' | N (character cells) | 'Npx' | 'N%' (default: '50%').
        height: Target height - 'auto' | N (character cells) | 'Npx' | 'N%' (default: 'auto').
        preserve_aspect_ratio: Preserve aspect ratio (default: True).
    """
    import base64

    b64 = base64.b64encode(png_data).decode()

    params = ["inline=1"]

    # Helper to format size value
    def format_size(value: SizeUnit) -> str:
        if value == "auto":
            return "auto"
        if isinstance(value, (int, float)):
            return str(int(value))  # bare number = character cells
        return str(value).lstrip()  # str like "200px" or "30%"

    w_str = format_size(width)
    h_str = format_size(height)

    if w_str != "auto":
        params.append(f"width={w_str}")
    if h_str != "auto":
        params.append(f"height={h_str}")
    if w_str != "auto" or h_str != "auto":
        params.append(f"preserveAspectRatio={'1' if preserve_aspect_ratio else '0'}")

    print(f"\033]1337;File={';'.join(params)}:{b64}\a", end="")
    print()
    
