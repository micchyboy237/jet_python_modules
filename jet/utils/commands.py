import json
import subprocess
from typing import Any
from jet.logger import logger
from jet.transformers.object import make_serializable


def copy_to_clipboard(text: Any):
    if not isinstance(text, str):
        text = make_serializable(text)
        text = json.dumps(text, indent=2, ensure_ascii=False)
    subprocess.run('pbcopy', input=text.encode(
        'utf-8'), check=True, env={'LANG': 'en_US.UTF-8'})
    logger.orange(f"Copied {len(text)} chars to clipboard!")


__all__ = [
    "copy_to_clipboard",
]
