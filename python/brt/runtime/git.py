# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
import pathlib
from typing import List


def update_gitignore(
    items: List[str], gitignore_path: pathlib.Path, file_suffix: str = None
) -> None:
    """update gitignore files with items

    Args:
        items (List): files to ignore
        gitignore (str): the gitignore file path
    """
    if items is None or len(items) == 0:
        return
    if gitignore_path.exists() is False:
        gitignore_path.touch(exist_ok=True)
    ignored_items = gitignore_path.read_text().splitlines()
    items_with_suffix = [item + file_suffix for item in items]
    ignoring_items = list(filter(lambda x: x not in ignored_items, items_with_suffix))
    if len(ignoring_items) == 0:
        return
    with gitignore_path.open("a") as f:
        f.write("\n".join(ignoring_items) + "\n")
