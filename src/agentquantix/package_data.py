from importlib.resources import files
from pathlib import Path, PurePosixPath


PACKAGE_ROOT = Path(files("agentquantix"))


def package_path(relative_path: str) -> Path:
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"invalid package data path: {relative_path}")
    return PACKAGE_ROOT.joinpath(*relative.parts)
