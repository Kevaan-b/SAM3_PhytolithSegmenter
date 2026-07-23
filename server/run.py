from __future__ import annotations

import os
import sys
from pathlib import Path


def nvidia_library_path() -> str:
    site = Path.home() / ".local" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "nvidia"
    paths = [str(path / "lib") for path in site.iterdir() if (path / "lib").is_dir()] if site.is_dir() else []
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
            "server.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
        environment,
    )
