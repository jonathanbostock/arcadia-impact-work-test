from __future__ import annotations

import importlib.util
from pathlib import Path


def resource_filename(package: str, resource: str) -> str:
    """Minimal pkg_resources shim for gym_sokoban.

    Returns a filesystem path to a packaged resource.
    """
    resource = resource.lstrip("/")
    spec = importlib.util.find_spec(package)
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError(f"No module named '{package}'")

    if spec.submodule_search_locations:
        base_path = Path(next(iter(spec.submodule_search_locations)))
    else:
        base_path = Path(spec.origin).parent

    return str(base_path / resource)
