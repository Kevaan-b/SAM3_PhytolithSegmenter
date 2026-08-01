from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file, save_file

FINGERPRINT_SCHEMA = 1
CACHE_SCHEMA = "samotator-embedding-v1"
EMBEDDING_KEYS = (
    "image_embeddings.0",
    "image_embeddings.1",
    "image_embeddings.2",
)
EXPECTED_SHAPES = (
    (1, 32, 288, 288),
    (1, 64, 144, 144),
    (1, 256, 72, 72),
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def cache_key(image_path: Path, model_fingerprint: str) -> str:
    digest = hashlib.sha256()
    digest.update(CACHE_SCHEMA.encode())
    digest.update(model_fingerprint.encode())
    digest.update(sha256_file(image_path).encode())
    return digest.hexdigest()


def model_fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256(CACHE_SCHEMA.encode())
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode())
        digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def save_embedding(
    destination: Path,
    arrays: dict[str, np.ndarray],
    original_size: tuple[int, int],
    reshaped_size: tuple[int, int],
) -> None:
    validate_embedding_arrays(arrays)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f".{os.getpid()}.tmp")
    payload = {
        **{key: np.asarray(arrays[key], dtype=np.float32) for key in EMBEDDING_KEYS},
        "original_size": np.asarray(original_size, dtype=np.int64),
        "reshaped_size": np.asarray(reshaped_size, dtype=np.int64),
    }
    save_file(payload, temporary, metadata={"schema": CACHE_SCHEMA})
    os.replace(temporary, destination)


def load_embedding(
    path: Path,
) -> tuple[dict[str, np.ndarray], tuple[int, int], tuple[int, int]]:
    arrays = load_file(path)
    embeddings = {key: arrays[key] for key in EMBEDDING_KEYS}
    validate_embedding_arrays(embeddings)
    original = tuple(int(value) for value in arrays["original_size"])
    reshaped = tuple(int(value) for value in arrays["reshaped_size"])
    if len(original) != 2 or len(reshaped) != 2:
        raise ValueError("Cached size metadata must contain height and width.")
    return embeddings, original, reshaped


def cache_file_is_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with safe_open(path, framework="np") as handle:
            if handle.metadata().get("schema") != CACHE_SCHEMA:
                return False
            keys = set(handle.keys())
            return set(EMBEDDING_KEYS).issubset(keys) and {
                "original_size",
                "reshaped_size",
            }.issubset(keys)
    except Exception:
        return False


def validate_embedding_arrays(arrays: dict[str, np.ndarray]) -> None:
    for key, shape in zip(EMBEDDING_KEYS, EXPECTED_SHAPES, strict=True):
        value = arrays.get(key)
        if value is None:
            raise ValueError(f"Missing embedding tensor {key}.")
        if value.dtype != np.float32 or value.shape != shape:
            raise ValueError(
                f"{key} must be float32 with shape {shape}, got "
                f"{value.dtype} {value.shape}."
            )


def materialize_external_data(model_dir: Path, cache_dir: Path) -> dict[str, Path]:
    """Return loadable ONNX paths, resolving Git LFS pointer sidecars if needed."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, Path] = {}
    for graph_name in ("vision_encoder_q4.onnx", "prompt_encoder_mask_decoder.onnx"):
        graph = model_dir / graph_name
        sidecar = model_dir / f"{graph_name}_data"
        if not graph.is_file() or not sidecar.is_file():
            raise FileNotFoundError(f"Missing model file or sidecar for {graph_name}.")

        sidecar_source = _resolve_lfs_pointer(sidecar)
        if sidecar_source == sidecar:
            resolved[graph_name] = graph
            continue

        target_graph = cache_dir / graph.name
        target_sidecar = cache_dir / sidecar.name
        shutil.copy2(graph, target_graph)
        temporary = target_sidecar.with_suffix(".tmp")
        temporary.unlink(missing_ok=True)
        try:
            os.link(sidecar_source, temporary)
        except OSError:
            shutil.copy2(sidecar_source, temporary)
        os.replace(temporary, target_sidecar)
        resolved[graph_name] = target_graph
    return resolved


def _resolve_lfs_pointer(path: Path) -> Path:
    if path.stat().st_size > 1024:
        return path
    text = path.read_text(errors="replace")
    if not text.startswith("version https://git-lfs.github.com/spec/v1"):
        return path
    values = dict(
        line.split(" ", 1) for line in text.splitlines() if " " in line
    )
    oid = values.get("oid", "").removeprefix("sha256:")
    expected_size = int(values.get("size", "0"))
    repository = next(
        (parent for parent in path.parents if (parent / ".git").exists()),
        None,
    )
    if not repository or len(oid) != 64:
        raise RuntimeError(f"{path} is an unresolved Git LFS pointer.")
    candidate = repository / ".git" / "lfs" / "objects" / oid[:2] / oid[2:4] / oid
    if not candidate.is_file() or candidate.stat().st_size != expected_size:
        raise RuntimeError(
            f"{path} is a Git LFS pointer but its {expected_size}-byte object is missing."
        )
    return candidate


T = TypeVar("T")


class ByteLru(Generic[T]):
    def __init__(self, budget_bytes: int):
        if budget_bytes <= 0:
            raise ValueError("LRU budget must be positive.")
        self.budget_bytes = budget_bytes
        self.total_bytes = 0
        self._items: OrderedDict[str, tuple[T, int]] = OrderedDict()

    def get(self, key: str) -> T | None:
        item = self._items.get(key)
        if item is None:
            return None
        self._items.move_to_end(key)
        return item[0]

    def put(self, key: str, value: T, size_bytes: int) -> list[T]:
        evicted: list[T] = []
        previous = self._items.pop(key, None)
        if previous:
            self.total_bytes -= previous[1]
        self._items[key] = (value, size_bytes)
        self.total_bytes += size_bytes
        while self.total_bytes > self.budget_bytes and len(self._items) > 1:
            _, (old_value, old_size) = self._items.popitem(last=False)
            self.total_bytes -= old_size
            evicted.append(old_value)
        return evicted

    def discard(self, key: str) -> T | None:
        item = self._items.pop(key, None)
        if item is None:
            return None
        self.total_bytes -= item[1]
        return item[0]

    def __len__(self) -> int:
        return len(self._items)


class ImageFingerprintIndex:
    """Persist cache keys so unchanged images are not rehashed at startup."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.RLock()
        self.document = self._load()
        self._save_timer: threading.Timer | None = None

    def cached_key(self, relative_path: str, image_path: Path, model: str) -> str | None:
        stat = image_path.stat()
        with self.lock:
            item = self.document["images"].get(relative_path)
            if not item or item.get("model") != model:
                return None
            if item.get("size") != stat.st_size or item.get("mtimeNs") != stat.st_mtime_ns:
                return None
            value = item.get("cacheKey")
            return value if isinstance(value, str) and len(value) == 64 else None

    def resolve_key(self, relative_path: str, image_path: Path, model: str) -> str:
        value = self.cached_key(relative_path, image_path, model)
        if value:
            return value
        value = cache_key(image_path, model)
        stat = image_path.stat()
        with self.lock:
            self.document["images"][relative_path] = {
                "size": stat.st_size, "mtimeNs": stat.st_mtime_ns,
                "model": model, "cacheKey": value,
            }
            self._schedule_save()
        return value

    def remove_missing(self, paths: set[str]) -> None:
        with self.lock:
            changed = False
            for value in list(self.document["images"]):
                if value not in paths:
                    del self.document["images"][value]
                    changed = True
            if changed:
                self._schedule_save()

    def flush(self) -> None:
        with self.lock:
            if self._save_timer:
                self._save_timer.cancel()
                self._save_timer = None
            self._save()

    def _schedule_save(self) -> None:
        if self._save_timer and self._save_timer.is_alive():
            return
        self._save_timer = threading.Timer(0.5, self._flush_timer)
        self._save_timer.daemon = True
        self._save_timer.start()

    def _flush_timer(self) -> None:
        with self.lock:
            self._save_timer = None
            self._save()

    def _load(self) -> dict:
        if self.path.is_file():
            try:
                value = json.loads(self.path.read_text(encoding="utf-8"))
                if value.get("schemaVersion") == FINGERPRINT_SCHEMA and isinstance(value.get("images"), dict):
                    return value
            except (OSError, ValueError):
                pass
        return {"schemaVersion": FINGERPRINT_SCHEMA, "images": {}}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(self.document, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, self.path)
