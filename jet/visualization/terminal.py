def display_iterm2_image(png_data: bytes):
    """Display PNG image inline in iTerm2 using imgcat protocol."""
    import base64
    b64 = base64.b64encode(png_data).decode()
    print(f"\033]1337;File=inline=1:{b64}\a", end="")
