from __future__ import annotations

import base64
import json
import os
import threading
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from .annotation_index import AnnotationIndex, CocoExporter
from .manifest import ImageRecord

PALETTE = ["#4094dc", "#e06b65", "#34a66f", "#9b72cf", "#e29a3b", "#26a6a1", "#d767a7", "#7f8f3f"]


class AnnotationStore:
    def __init__(self, data_root: Path, get_record: Callable[[str], ImageRecord], cache_root: Path | None = None) -> None:
        self.data_root = data_root.resolve()
        self.get_record = get_record
        self.metadata_root = self.data_root / "metadata"
        self.output_root = self.data_root / "annotations"
        self.internal_root = self.data_root / ".samotator"
        self.draft_root = self.internal_root / "annotations"
        self.categories_path = self.metadata_root / "categories.json"
        self.ids_path = self.internal_root / "ids.json"
        self.lock = threading.RLock()
        cache = cache_root or self.data_root.parent / ".samotator-cache"
        self.preview_root = cache / "previews"
        self.preview_lock = threading.RLock()
        self.index = AnnotationIndex(cache / "annotations.sqlite3", self.draft_root)
        self.exporter = CocoExporter(self.index, self.output_root, lambda: self._read_categories()["categories"])

    def initialize(self) -> None:
        changed_splits = self.index.reconcile()
        for split in changed_splits:
            self.exporter.schedule(split)

    def set_interactive(self, active: bool) -> None:
        self.exporter.set_interactive(active)

    def shutdown(self) -> None:
        self.exporter.stop()

    def flush_exports(self) -> None:
        self.exporter.flush()

    def categories(self) -> dict:
        with self.lock:
            return self._read_categories()

    def statistics(self) -> dict:
        """Return indexed saved-instance counts without expanding any masks."""
        return self.index.statistics(self._read_categories()["categories"])

    def statistics_previews(self, category_id: int, page: int = 1) -> dict:
        category = self._category(self._read_categories(), category_id)
        result = self.index.previews(category_id, page, 8)
        for item in result["previews"]:
            version = self._preview_version(item["imageId"], category_id, category["color"])
            item["previewUrl"] = f"/api/statistics/previews/{item['imageId']}?category_id={category_id}&v={version}"
        return result

    def _preview_version(self, image_id: str, category_id: int, color: str) -> str:
        import hashlib
        record = self.get_record(image_id)
        stat = record.absolute_path.stat()
        base = self.index.image_version(image_id, category_id, color)
        return hashlib.sha256(f"{base}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()[:16]

    def preview_image(self, image_id: str, category_id: int | None = None) -> bytes:
        with self.preview_lock:
            return self._preview_image_unlocked(image_id, category_id)

    def _preview_image_unlocked(self, image_id: str, category_id: int | None = None) -> bytes:
        record = self.get_record(image_id)
        draft = self.load_image(image_id)
        if not draft["layers"]:
            raise ValueError("No saved annotations exist for this image.")
        categories = {int(item["id"]): item for item in self._read_categories()["categories"]}
        selected = category_id if category_id is not None else int(draft["layers"][0]["categoryId"])
        category = categories.get(selected)
        if not category:
            raise ValueError("Unknown preview category.")
        version = self._preview_version(image_id, selected, category["color"])
        destination = self.preview_root / f"{image_id}-{selected}-{version}.webp"
        if destination.is_file():
            return destination.read_bytes()
        with self.preview_lock:
            if destination.is_file():
                return destination.read_bytes()
            destination.parent.mkdir(parents=True, exist_ok=True)
            stat = record.absolute_path.stat()
            source_key = f"{image_id}-{stat.st_size}-{stat.st_mtime_ns}.webp"
            source_path = self.preview_root / "sources" / source_key
            if source_path.is_file():
                with Image.open(source_path) as cached_source:
                    image = cached_source.convert("RGBA")
            else:
                with Image.open(record.absolute_path) as source:
                    image = source.convert("RGBA")
                    image.thumbnail((320, 320))
                source_path.parent.mkdir(parents=True, exist_ok=True)
                image.convert("RGB").save(source_path, format="WEBP", quality=82, method=4)
        width, height = int(draft["width"]), int(draft["height"])
        rgb = tuple(int(category["color"][index:index + 2], 16) for index in (1, 3, 5))
        for layer in draft["layers"]:
            if int(layer["categoryId"]) != selected:
                continue
            mask = Image.fromarray(self._unpack(layer["effectiveMask"], width, height).astype(np.uint8) * 120, mode="L")
            mask = mask.resize(image.size, Image.Resampling.NEAREST)
            overlay = Image.new("RGBA", image.size, (*rgb, 0))
            overlay.putalpha(mask)
            image = Image.alpha_composite(image, overlay)
        output = BytesIO()
        image.convert("RGB").save(output, format="WEBP", quality=82, method=4)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        temporary.write_bytes(output.getvalue())
        os.replace(temporary, destination)
        return output.getvalue()

    def add_category(self, name: str, supercategory: str = "phytolith", color: str | None = None) -> dict:
        with self.lock:
            document = self._read_categories()
            normalized = self._validate_name(name, document)
            identifier = int(document["next_category_id"])
            category = {
                "id": identifier,
                "name": normalized,
                "supercategory": self._validate_supercategory(supercategory),
                "color": self._validate_color(color or PALETTE[(identifier - 1) % len(PALETTE)]),
                "active": True,
            }
            document["next_category_id"] = identifier + 1
            document["categories"].append(category)
            self._write_json(self.categories_path, document)
            for split in self.index.splits():
                self.exporter.schedule(split)
            return category

    def update_category(self, identifier: int, changes: dict) -> dict:
        with self.lock:
            document = self._read_categories()
            category = self._category(document, identifier)
            if "name" in changes:
                category["name"] = self._validate_name(str(changes["name"]), document, identifier)
            if "supercategory" in changes:
                category["supercategory"] = self._validate_supercategory(str(changes["supercategory"]))
            if "color" in changes:
                category["color"] = self._validate_color(str(changes["color"]))
            if "active" in changes:
                category["active"] = bool(changes["active"])
            self._write_json(self.categories_path, document)
            for split in self.index.splits():
                self.exporter.schedule(split)
            return category

    def archive_category(self, identifier: int) -> dict:
        return self.update_category(identifier, {"active": False})

    def load_image(self, image_id: str) -> dict:
        record = self.get_record(image_id)
        path = self.draft_root / f"{image_id}.json"
        with self.lock:
            if not path.exists():
                return {"imageId": image_id, "layers": [], "latestMaskLayerId": None, "preventOverlap": False}
            draft = self._read_json(path)
            if draft.get("file_name") != record.relative_path:
                raise ValueError("Saved annotation path does not match the selected image.")
            return draft

    def save_image(self, image_id: str, width: int, height: int, layers: list[dict], latest_layer_id: str | None, prevent_overlap: bool) -> dict:
        record = self.get_record(image_id)
        with Image.open(record.absolute_path) as image:
            actual_width, actual_height = image.size
        if (width, height) != (actual_width, actual_height):
            raise ValueError("Annotation dimensions do not match the image.")
        expected_bytes = (width * height + 7) // 8
        categories = self._read_categories()
        category_ids = {int(item["id"]) for item in categories["categories"]}
        ids = self._read_ids()
        image_number = self._image_number(ids, record.relative_path)
        seen: set[str] = set()
        saved_layers: list[dict] = []
        for layer in layers:
            layer_id = str(layer.get("layerId", ""))
            if not layer_id or layer_id in seen:
                raise ValueError("Every annotation layer must have a unique ID.")
            seen.add(layer_id)
            category_id = int(layer.get("categoryId", 0))
            if category_id not in category_ids:
                raise ValueError(f"Unknown category ID {category_id}.")
            raw = self._decode_mask(str(layer.get("rawMask", "")), expected_bytes)
            effective = self._decode_mask(str(layer.get("effectiveMask", "")), expected_bytes)
            annotation_id = self._annotation_number(ids, layer_id)
            saved_layers.append({
                "layerId": layer_id,
                "annotationId": annotation_id,
                "categoryId": category_id,
                "rawMask": base64.b64encode(raw).decode("ascii"),
                "effectiveMask": base64.b64encode(effective).decode("ascii"),
            })
        draft = {
            "schemaVersion": 1,
            "imageId": image_id,
            "imageNumber": image_number,
            "file_name": record.relative_path,
            "width": width,
            "height": height,
            "latestMaskLayerId": latest_layer_id,
            "preventOverlap": bool(prevent_overlap),
            "layers": saved_layers,
            "savedAt": datetime.now(UTC).isoformat(),
        }
        with self.lock:
            self._write_json(self.ids_path, ids)
            draft_path = self.draft_root / f"{image_id}.json"
            self._write_json(draft_path, draft)
            self.index.upsert(draft)
            self.exporter.schedule(self._split(record.relative_path))
        nonempty = sum(self._mask_area(item["effectiveMask"], width, height) > 0 for item in saved_layers)
        return {"imageId": image_id, "savedLayers": nonempty, "emptyLayers": len(saved_layers) - nonempty, "savedAt": draft["savedAt"]}

    def _read_categories(self) -> dict:
        if not self.categories_path.exists():
            document = {
                "schema_version": 1,
                "next_category_id": 2,
                "categories": [{"id": 1, "name": "object", "supercategory": "phytolith", "color": PALETTE[0], "active": True}],
            }
            self._write_json(self.categories_path, document)
            return document
        document = self._read_json(self.categories_path)
        if document.get("schema_version") != 1 or not isinstance(document.get("categories"), list):
            raise ValueError("metadata/categories.json has an unsupported format.")
        return document

    def _read_ids(self) -> dict:
        if self.ids_path.exists():
            return self._read_json(self.ids_path)
        return {"schemaVersion": 1, "nextImageId": 1, "nextAnnotationId": 1, "images": {}, "annotations": {}}

    def _image_number(self, ids: dict, path: str) -> int:
        if path not in ids["images"]:
            ids["images"][path] = ids["nextImageId"]
            ids["nextImageId"] += 1
        return int(ids["images"][path])

    def _annotation_number(self, ids: dict, layer_id: str) -> int:
        if layer_id not in ids["annotations"]:
            ids["annotations"][layer_id] = ids["nextAnnotationId"]
            ids["nextAnnotationId"] += 1
        return int(ids["annotations"][layer_id])

    @staticmethod
    def _split(path: str) -> str:
        parts = Path(path).parts
        if parts and parts[0] == "images":
            parts = parts[1:]
        return parts[0] if parts and parts[0] in {"train", "val", "test"} else "default"

    @staticmethod
    def _decode_mask(value: str, expected: int) -> bytes:
        try:
            decoded = base64.b64decode(value, validate=True)
        except Exception as error:
            raise ValueError("Invalid base64 mask.") from error
        if len(decoded) != expected:
            raise ValueError("Mask byte length does not match the image dimensions.")
        return decoded

    @staticmethod
    def _unpack(value: str, width: int, height: int) -> np.ndarray:
        packed = np.frombuffer(base64.b64decode(value), dtype=np.uint8)
        return np.unpackbits(packed, bitorder="little")[: width * height].reshape(height, width).astype(bool)

    def _mask_area(self, value: str, width: int, height: int) -> int:
        return int(self._unpack(value, width, height).sum())

    @staticmethod
    def _validate_color(color: str) -> str:
        import re
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
            raise ValueError("Category colors must use #RRGGBB format.")
        return color.lower()

    @staticmethod
    def _validate_supercategory(value: str) -> str:
        value = value.strip()[:80]
        if not value:
            raise ValueError("Supercategory cannot be empty.")
        return value

    @staticmethod
    def _validate_name(name: str, document: dict, ignore_id: int | None = None) -> str:
        normalized = name.strip()[:80]
        if not normalized:
            raise ValueError("Category name cannot be empty.")
        if any(item["name"].casefold() == normalized.casefold() and int(item["id"]) != ignore_id for item in document["categories"]):
            raise ValueError("Category names must be unique.")
        return normalized

    @staticmethod
    def _category(document: dict, identifier: int) -> dict:
        for category in document["categories"]:
            if int(category["id"]) == identifier:
                return category
        raise KeyError("Unknown category ID.")

    @staticmethod
    def _read_json(path: Path) -> dict:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _write_json(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
