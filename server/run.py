from __future__ import annotations

import os
import sys
from pathlib import Path


def nvidia_library_path() -> str:
    # Pip can install NVIDIA libraries in a virtual environment, a conda
    # environment, or the user site. Follow this interpreter's import paths.
    paths: list[str] = []
    for entry in sys.path:
        site = Path(entry) / "nvidia" if entry else None
        if site and site.is_dir():
            for package in sorted(site.iterdir()):
                lib = package / "lib"
                if lib.is_dir() and str(lib) not in paths:
                    paths.append(str(lib))
    existing = os.environ.get("LD_LIBRARY_PATH")
    if existing:
        paths.append(existing)
    return ":".join(paths)


if __name__ == "__main__":
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = nvidia_library_path()
    os.execvpe(
        sys.executable,
        [
            sys.executable,
            "-m",
            "uvicorn",
            "server.gateway:app",
            "--host",
            os.environ.get("SAMOTATOR_HOST", "0.0.0.0"),
            "--port",
            os.environ.get("SAMOTATOR_PORT", "8000"),
        ],
        environment,
    )
