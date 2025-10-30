import os
from jet.utils.inspect_utils import get_entry_file_dir, get_entry_file_name


DEFAULT_LOG_DIR = os.path.expanduser(
    f"{get_entry_file_dir()}/generated/{os.path.splitext(get_entry_file_name())[0]}"
)
