from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np
from pycocotools import mask as coco_mask

INDEX_SCHEMA = 1


class AnnotationIndex:
    """Rebuildable metadata index for fast counts, previews, and COCO export."""

    def __init__(self, path: Path, draft_root: Path) -> None:
        self.path = path
        self.draft_root = draft_root
        self.lock = threading.RLock()
        self.ready = False
        try:
            self._initialize_schema()
        except sqlite3.DatabaseError:
            for candidate in (self.path, Path(str(self.path) + "-wal"), Path(str(self.path) + "-shm")):
                candidate.unlink(missing_ok=True)
            self._initialize_schema()

    def rebuild(self) -> None:
        with self.lock, self._connect() as connection:
            connection.execute("DELETE FROM annotations")
            connection.execute("DELETE FROM images")
        self.reconcile()

    def reconcile(self) -> set[str]:
        changed = False
        paths = sorted(self.draft_root.glob("*.json")) if self.draft_root.exists() else []
        known = self._fingerprints()
        present: set[str] = set()
        original_ids = set(known)
        for path in paths:
            image_id = path.stem
            present.add(image_id)
            stat = path.stat()
            fingerprint = (stat.st_mtime_ns, stat.st_size)
            if known.get(image_id) == fingerprint:
                continue
            try:
                self.upsert(json.loads(path.read_text(encoding="utf-8")), fingerprint)
                changed = True
            except (OSError, ValueError, KeyError, TypeError):
                continue
        changed = changed or bool(original_ids - present)
        with self.lock, self._connect() as connection:
            if present:
                placeholders = ",".join("?" for _ in present)
                connection.execute(f"DELETE FROM images WHERE image_id NOT IN ({placeholders})", tuple(present))
            else:
                connection.execute("DELETE FROM images")
        self.ready = True
        return set(self.splits()) if changed else set()

    def upsert(self, draft: dict, fingerprint: tuple[int, int] | None = None) -> None:
        image_id = str(draft["imageId"])
        width, height = int(draft["width"]), int(draft["height"])
        if fingerprint is None:
            stat = (self.draft_root / f"{image_id}.json").stat()
            fingerprint = (stat.st_mtime_ns, stat.st_size)
        annotations: list[tuple] = []
        for layer in draft.get("layers", []):
            binary = self._unpack(layer["effectiveMask"], width, height)
            if not binary.any():
                continue
            encoded = coco_mask.encode(np.asfortranarray(binary.astype(np.uint8)))
            bbox = [float(value) for value in coco_mask.toBbox(encoded).tolist()]
            annotations.append((int(layer["annotationId"]), image_id, str(layer["layerId"]),
                int(layer["categoryId"]), int(coco_mask.area(encoded)), *bbox,
                encoded["counts"].decode("ascii")))
        with self.lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO images(image_id,image_number,file_name,width,height,split,saved_at,draft_mtime_ns,draft_size)
                VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(image_id) DO UPDATE SET
                image_number=excluded.image_number,file_name=excluded.file_name,width=excluded.width,
                height=excluded.height,split=excluded.split,saved_at=excluded.saved_at,
                draft_mtime_ns=excluded.draft_mtime_ns,draft_size=excluded.draft_size""",
                (image_id, int(draft["imageNumber"]), draft["file_name"], width, height,
                 self.split_name(draft["file_name"]), draft.get("savedAt", ""), *fingerprint))
            connection.execute("DELETE FROM annotations WHERE image_id=?", (image_id,))
            connection.executemany("""INSERT INTO annotations(annotation_id,image_id,layer_id,category_id,area,
                bbox_x,bbox_y,bbox_w,bbox_h,rle_counts) VALUES(?,?,?,?,?,?,?,?,?,?)""", annotations)

    def statistics(self, categories: list[dict]) -> dict:
        with self.lock, self._connect() as connection:
            counts = {int(row[0]): int(row[1]) for row in connection.execute(
                "SELECT category_id,COUNT(*) FROM annotations GROUP BY category_id")}
        return {"totalAnnotations": sum(counts.values()), "classes": [
            {**category, "annotationCount": counts.get(int(category["id"]), 0)} for category in categories]}

    def previews(self, category_id: int, limit: int) -> list[dict]:
        with self.lock, self._connect() as connection:
            rows = connection.execute("""SELECT i.image_id,i.file_name,i.saved_at,COUNT(a.annotation_id)
                FROM images i JOIN annotations a ON a.image_id=i.image_id WHERE a.category_id=?
                GROUP BY i.image_id ORDER BY i.saved_at DESC,i.image_id LIMIT ?""",
                (category_id, min(max(limit, 1), 8))).fetchall()
        return [{"imageId": row[0], "fileName": row[1], "savedAt": row[2],
                 "annotationCount": int(row[3])} for row in rows]

    def splits(self) -> list[str]:
        with self.lock, self._connect() as connection:
            return [str(row[0]) for row in connection.execute("SELECT DISTINCT split FROM images")]

    def coco_document(self, split: str, categories: list[dict]) -> dict:
        with self.lock, self._connect() as connection:
            images = connection.execute("SELECT image_id,image_number,file_name,width,height FROM images WHERE split=? ORDER BY image_number", (split,)).fetchall()
            annotations = connection.execute("""SELECT a.annotation_id,i.image_number,a.category_id,a.bbox_x,a.bbox_y,
                a.bbox_w,a.bbox_h,a.area,a.rle_counts,i.height,i.width FROM annotations a
                JOIN images i ON i.image_id=a.image_id WHERE i.split=? ORDER BY a.annotation_id""", (split,)).fetchall()
        return {"info": {"description": "Samotator COCO instance segmentation dataset", "version": "1.0"},
            "licenses": [],
            "images": [{"id": int(row[1]), "file_name": self.coco_file_name(row[2]),
                        "width": int(row[3]), "height": int(row[4])} for row in images],
            "annotations": [{"id": int(row[0]), "image_id": int(row[1]), "category_id": int(row[2]),
                "bbox": [float(value) for value in row[3:7]], "area": int(row[7]),
                "segmentation": {"size": [int(row[9]), int(row[10])], "counts": row[8]},
                "iscrowd": 0} for row in annotations],
            "categories": [{"id": int(item["id"]), "name": item["name"],
                "supercategory": item.get("supercategory", "")} for item in sorted(categories, key=lambda item: int(item["id"]))]}

    def image_version(self, image_id: str, category_id: int, color: str) -> str:
        with self.lock, self._connect() as connection:
            row = connection.execute("""SELECT i.saved_at,COALESCE(GROUP_CONCAT(a.annotation_id || ':' || a.area),'')
                FROM images i LEFT JOIN annotations a ON a.image_id=i.image_id AND a.category_id=?
                WHERE i.image_id=? GROUP BY i.image_id""", (category_id, image_id)).fetchone()
        if not row:
            raise KeyError(image_id)
        return hashlib.sha256(f"preview-v2:{row[0]}:{row[1]}:{color}".encode()).hexdigest()[:16]

    def _fingerprints(self) -> dict[str, tuple[int, int]]:
        with self.lock, self._connect() as connection:
            return {str(row[0]): (int(row[1]), int(row[2])) for row in connection.execute(
                "SELECT image_id,draft_mtime_ns,draft_size FROM images")}

    def _initialize_schema(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
            version = connection.execute("SELECT value FROM metadata WHERE key='schema'").fetchone()
            if version and int(version[0]) != INDEX_SCHEMA:
                connection.execute("DROP TABLE IF EXISTS annotations")
                connection.execute("DROP TABLE IF EXISTS images")
            connection.execute("""CREATE TABLE IF NOT EXISTS images(image_id TEXT PRIMARY KEY,
                image_number INTEGER NOT NULL,file_name TEXT NOT NULL,width INTEGER NOT NULL,height INTEGER NOT NULL,
                split TEXT NOT NULL,saved_at TEXT NOT NULL,draft_mtime_ns INTEGER NOT NULL,draft_size INTEGER NOT NULL)""")
            connection.execute("""CREATE TABLE IF NOT EXISTS annotations(annotation_id INTEGER PRIMARY KEY,
                image_id TEXT NOT NULL REFERENCES images(image_id) ON DELETE CASCADE,layer_id TEXT NOT NULL,
                category_id INTEGER NOT NULL,area INTEGER NOT NULL,bbox_x REAL NOT NULL,bbox_y REAL NOT NULL,
                bbox_w REAL NOT NULL,bbox_h REAL NOT NULL,rle_counts TEXT NOT NULL)""")
            connection.execute("CREATE INDEX IF NOT EXISTS annotation_category ON annotations(category_id,image_id)")
            connection.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES('schema',?)", (str(INDEX_SCHEMA),))

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @staticmethod
    def _unpack(value: str, width: int, height: int) -> np.ndarray:
        packed = np.frombuffer(base64.b64decode(value), dtype=np.uint8)
        return np.unpackbits(packed, bitorder="little")[:width * height].reshape(height, width).astype(bool)

    @staticmethod
    def split_name(path: str) -> str:
        parts = Path(path).parts
        if parts and parts[0] == "images": parts = parts[1:]
        return parts[0] if parts and parts[0] in {"train", "val", "test"} else "default"

    @staticmethod
    def coco_file_name(path: str) -> str:
        parts = Path(path).parts
        return Path(*parts[1:]).as_posix() if parts and parts[0] == "images" else Path(*parts).as_posix()


class CocoExporter:
    def __init__(self, index: AnnotationIndex, output_root: Path, categories: Callable[[], list[dict]]) -> None:
        self.index, self.output_root, self.categories = index, output_root, categories
        self.condition = threading.Condition()
        self.pending: set[str] = set()
        self.interactive = False
        self.stopped = False
        self.exporting = False
        self.thread = threading.Thread(target=self._run, daemon=True, name="samotator-coco")
        self.thread.start()

    def schedule(self, split: str) -> None:
        with self.condition:
            self.pending.add(split)
            self.condition.notify_all()

    def set_interactive(self, active: bool) -> None:
        with self.condition:
            self.interactive = active
            self.condition.notify_all()

    def flush(self, timeout: float = 10) -> None:
        deadline = time.monotonic() + timeout
        with self.condition:
            self.interactive = False
            self.condition.notify_all()
            while (self.pending or self.exporting) and time.monotonic() < deadline:
                self.condition.wait(timeout=max(0, min(0.1, deadline - time.monotonic())))

    def stop(self) -> None:
        self.flush()
        with self.condition:
            self.stopped = True
            self.condition.notify_all()
        self.thread.join(timeout=5)

    def _run(self) -> None:
        while True:
            with self.condition:
                while (not self.pending or self.interactive) and not self.stopped: self.condition.wait()
                if self.stopped: return
                self.condition.wait(timeout=1.0)
                if self.interactive: continue
                splits, self.pending = sorted(self.pending), set()
                self.exporting = True
            try:
                categories = self.categories()
                for split in splits:
                    name = "instances.json" if split == "default" else f"instances_{split}.json"
                    self._write_json(self.output_root / name, self.index.coco_document(split, categories))
            except Exception:
                with self.condition:
                    self.pending.update(splits)
                time.sleep(1)
            finally:
                with self.condition:
                    self.exporting = False
                    self.condition.notify_all()

    @staticmethod
    def _write_json(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{threading.get_ident()}.tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        temporary.replace(path)


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Rebuild Samotator's derived annotation index.")
    parser.add_argument("command", choices=["rebuild"])
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    data_root = args.project_root.resolve() / "data"
    index = AnnotationIndex(args.project_root.resolve() / ".samotator-cache" / "annotations.sqlite3",
                            data_root / ".samotator" / "annotations")
    index.rebuild()
    print(f"Indexed {index.statistics([])['totalAnnotations']} annotations.")


if __name__ == "__main__":
    _main()
