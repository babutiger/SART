"""Compatibility helper for archived scripts that still bootstrap `sys.path`."""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = ["add_parent_dirs_to_sys_path"]


def add_parent_dirs_to_sys_path(start_path: str | os.PathLike[str] | None = None):
    """Append a path and each of its parents to `sys.path` once, from leaf to root."""
    current_path = Path(start_path or os.getcwd()).resolve()
    added_paths = []

    for parent_path in [current_path, *current_path.parents]:
        parent_str = str(parent_path)
        if parent_str not in sys.path:
            sys.path.append(parent_str)
            added_paths.append(parent_str)

    return added_paths


if __name__ == "__main__":
    add_parent_dirs_to_sys_path()
