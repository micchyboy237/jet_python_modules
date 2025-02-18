import subprocess
from jet.logger import logger


def copy_to_clipboard(text: str):
    subprocess.run('pbcopy', input=text.encode(
        'utf-8'), check=True, env={'LANG': 'en_US.UTF-8'})
    logger.orange(f"Copied {len(text)} chars to clipboard!")


__all__ = [
    "copy_to_clipboard",
]
