from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

IMAGE_EXTENSIONS = {
    ".avif", ".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"
}


@dataclass(frozen=True)
class ImageRecord:
    id: str
    name: str
    relative_path: str
    absolute_path: Path
    folder_path: str


def image_id(relative_path: str) -> str:
    return hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:24]


def scan_data(data_root: Path) -> tuple[dict, dict[str, ImageRecord]]:
    root = data_root.resolve()
    records: dict[str, ImageRecord] = {}

    def scan(folder: Path, relative: Path, name: str) -> dict:
        folders: list[dict] = []
        images: list[dict] = []
        if not folder.exists():
            return {"name": name, "path": relative.as_posix() if relative.parts else "", "folders": [], "images": []}
        for entry in sorted(folder.iterdir(), key=lambda item: natural_key(item.name)):
            if entry.name.startswith(".") or entry.is_symlink():
                continue
            child_relative = relative / entry.name
            if entry.is_dir():
                folders.append(scan(entry, child_relative, entry.name))
            elif entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS:
                relative_text = child_relative.as_posix()
                identifier = image_id(relative_text)
                if identifier in records:
                    raise RuntimeError(f"Image ID collision for {relative_text}.")
                record = ImageRecord(
                    id=identifier,
                    name=entry.name,
                    relative_path=relative_text,
                    absolute_path=entry.resolve(),
                    folder_path=relative.as_posix() if relative.parts else "",
                )
                if not record.absolute_path.is_relative_to(root):
                    raise RuntimeError(f"Image escaped data root: {relative_text}")
                records[identifier] = record
                images.append(
                    {
                        "id": identifier,
                        "name": entry.name,
                        "path": relative_text,
                        "url": "/data/" + "/".join(quote(part, safe="") for part in child_relative.parts),
                        "cacheState": "missing",
                    }
                )
        return {
            "name": name,
            "path": relative.as_posix() if relative.parts else "",
            "folders": folders,
            "images": images,
        }

    return scan(root, Path(), "Data"), records


def natural_key(value: str) -> tuple:
    import re
    return tuple(int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", value))


def set_cache_states(tree: dict, states: dict[str, str]) -> None:
    for image in tree["images"]:
        image["cacheState"] = states.get(image["id"], "missing")
    for folder in tree["folders"]:
        set_cache_states(folder, states)
