from pathlib import Path
from typing import List, Optional, Union


def get_most_recently_edited_file(directory: Union[str, Path]) -> Optional[Path]:
    """Returns the path of the most recently edited file in the given directory.

    Args:
        directory (str): The path to the directory to search.

    Returns:
        Optional[Path]: The path of the most recently edited file in the given directory,
            or None if there are no files in the directory.
    """

    files: List[Path] = [f for f in Path(directory).glob("*.ckpt") if f.is_file()]
    if not files:
        return None
    most_recently_edited_file = max(files, key=lambda f: f.stat().st_mtime)

    return most_recently_edited_file
