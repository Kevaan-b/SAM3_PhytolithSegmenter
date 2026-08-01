from __future__ import annotations

import asyncio
import concurrent.futures
import heapq
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

from .cache import (
    EMBEDDING_KEYS,
    ByteLru,
    cache_file_is_valid,
    ImageFingerprintIndex,
    load_embedding,
    materialize_external_data,
    model_fingerprint,
    save_embedding,
)
from .manifest import ImageRecord, scan_data, set_cache_states

INPUT_SIZE = (1008, 1008)
MASK_THRESHOLD = 0.0


@dataclass
class GpuEmbedding:
    tensors: dict[str, ort.OrtValue]
    original_size: tuple[int, int]
    reshaped_size: tuple[int, int]
    size_bytes: int


@dataclass(order=True)
class QueuedJob:
    priority: int
    sequence: int
    version: int = field(compare=False)
    key: str | None = field(compare=False)
    future: concurrent.futures.Future = field(compare=False)
    callback: Callable = field(compare=False)


class PriorityGpuExecutor:
    """Single-GPU executor with keyed decrease/increase-priority updates."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._jobs: list[QueuedJob] = []
        self._pending: dict[str, QueuedJob] = {}
        self._sequence = 0
        self._unkeyed = 0
        self._stopped = False
        self._interactive = False
        self._foreground_until = 0.0
        self._current_job: str | None = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="sam3-gpu")
        self._thread.start()

    def submit(self, priority: int, callback: Callable, key: str | None = None):
        with self._condition:
            if priority == 0:
                self._foreground_until = time.monotonic() + 0.25
            if key is not None and key in self._pending:
                previous = self._pending[key]
                self._sequence += 1
                job = QueuedJob(priority, self._sequence, previous.version + 1, key, previous.future, callback)
                self._pending[key] = job
                heapq.heappush(self._jobs, job)
                self._condition.notify_all()
                return previous.future
            future: concurrent.futures.Future = concurrent.futures.Future()
            self._sequence += 1
            job = QueuedJob(priority, self._sequence, 1, key, future, callback)
            if key is not None:
                self._pending[key] = job
            else:
                self._unkeyed += 1
            heapq.heappush(self._jobs, job)
            self._condition.notify_all()
            return future

    def set_interactive(self, active: bool) -> None:
        with self._condition:
            self._interactive = active
            self._condition.notify_all()

    @property
    def queue_depth(self) -> int:
        with self._condition:
            return len(self._pending) + self._unkeyed

    @property
    def current_job(self) -> str | None:
        with self._condition:
            return self._current_job

    @property
    def background_paused(self) -> bool:
        with self._condition:
            return self._interactive or time.monotonic() < self._foreground_until

    def stop(self) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        self._thread.join(timeout=5)

    def _next_job(self) -> QueuedJob | None:
        while self._jobs:
            job = heapq.heappop(self._jobs)
            if job.key is not None and self._pending.get(job.key) is not job:
                continue
            if job.future.done():
                if job.key is not None: self._pending.pop(job.key, None)
                continue
            return job
        return None

    def _run(self) -> None:
        while True:
            with self._condition:
                job = self._next_job()
                while job is None and not self._stopped:
                    self._condition.wait()
                    job = self._next_job()
                if self._stopped: return
                assert job is not None
                paused_for = max(0.0, self._foreground_until - time.monotonic())
                if job.priority >= 30 and (self._interactive or paused_for > 0):
                    heapq.heappush(self._jobs, job)
                    self._condition.wait(timeout=0.1 if self._interactive else paused_for)
                    continue
                if job.key is not None:
                    self._pending.pop(job.key, None)
                else:
                    self._unkeyed -= 1
                self._current_job = job.key or "foreground"
            try:
                job.future.set_result(job.callback())
            except BaseException as error:
                job.future.set_exception(error)
            finally:
                with self._condition:
                    self._current_job = None
                    self._condition.notify_all()


class SamEngine:
    def __init__(self, project_root: Path, gpu_budget_gib: float = 16) -> None:
        self.project_root = project_root.resolve()
        self.data_root = self.project_root / "data"
        self.cache_root = self.project_root / ".samotator-cache"
        self.embedding_root = self.cache_root / "embeddings"
        self.fingerprints = ImageFingerprintIndex(self.cache_root / "image-fingerprints.json")
        self.gpu_cache = ByteLru[GpuEmbedding](int(gpu_budget_gib * 1024**3))
        self.executor = PriorityGpuExecutor()
        self.lock = threading.RLock()
        self.ready = False
        self.error: str | None = None
        self.device = "initializing"
        self.tree, self.records = scan_data(self.data_root)
        self.states: dict[str, str] = {identifier: "missing" for identifier in self.records}
        self.keys: dict[str, str] = {}
        self.cache_paths: dict[str, Path] = {}
        self.model_fingerprint = ""
        self.encode_futures: dict[str, concurrent.futures.Future] = {}
        self.encoder: ort.InferenceSession | None = None
        self.decoder: ort.InferenceSession | None = None
        self.last_encode_ms: float | None = None
        self.last_decode_ms: float | None = None
        self.active_folder = ""
        self.selected_image: str | None = None
        self.visited_folders: set[str] = set()
        self.promote_ids: set[str] = set()
        self.folder_order: list[str] = []
        self.preprocess_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="sam3-preprocess")
        self.preprocess_futures: dict[str, concurrent.futures.Future] = {}
        self.cache_writer = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="sam3-cache-writer")
        self.write_slots = threading.Semaphore(2)
        self.pending_arrays: dict[str, tuple[dict[str, np.ndarray], tuple[int, int], tuple[int, int]]] = {}

    def initialize(self) -> None:
        try:
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("ONNX Runtime CUDAExecutionProvider is unavailable.")
            paths = materialize_external_data(
                self.project_root / "sam3-q4" / "onnx",
                self.cache_root / "model",
            )
            fingerprint_paths = []
            for path in paths.values():
                fingerprint_paths.extend([path, path.with_name(f"{path.name}_data")])
            self.model_fingerprint = model_fingerprint(
                [
                    *fingerprint_paths,
                    self.project_root / "sam3-q4" / "config.json",
                    self.project_root / "sam3-q4" / "preprocessor_config.json",
                ]
            )
            self.embedding_root.mkdir(parents=True, exist_ok=True)
            for identifier, record in self.records.items():
                key = self.fingerprints.cached_key(record.relative_path, record.absolute_path, self.model_fingerprint)
                if key:
                    path = self.embedding_root / f"{key}.safetensors"
                    self.keys[identifier] = key
                    self.cache_paths[identifier] = path
                    self.states[identifier] = "ready" if cache_file_is_valid(path) else "missing"
            self.fingerprints.remove_missing({record.relative_path for record in self.records.values()})

            providers = ["CUDAExecutionProvider"]
            self.encoder = ort.InferenceSession(
                str(paths["vision_encoder_q4.onnx"]), providers=providers
            )
            self.decoder = ort.InferenceSession(
                str(paths["prompt_encoder_mask_decoder.onnx"]), providers=providers
            )
            if self.encoder.get_providers()[0] != "CUDAExecutionProvider":
                raise RuntimeError("SAM3 encoder did not initialize on CUDA.")
            self.device = "CUDAExecutionProvider"
            self.ready = True
        except BaseException as error:
            self.error = str(error)
            self.device = "error"

    def refresh_manifest(self) -> dict:
        tree, records = scan_data(self.data_root)
        with self.lock:
            removed = set(self.records) - set(records)
            for identifier in removed:
                old_key = self.keys.pop(identifier, None)
                self.states.pop(identifier, None)
                self.cache_paths.pop(identifier, None)
                self.encode_futures.pop(identifier, None)
                self.preprocess_futures.pop(identifier, None)
                if old_key: self.gpu_cache.discard(old_key)
            for identifier, record in records.items():
                self.states.setdefault(identifier, "missing")
                if not self.model_fingerprint:
                    continue
                key = self.fingerprints.cached_key(record.relative_path, record.absolute_path, self.model_fingerprint)
                previous_key = self.keys.get(identifier)
                if key == previous_key:
                    continue
                if previous_key:
                    self.gpu_cache.discard(previous_key)
                self.keys.pop(identifier, None)
                self.cache_paths.pop(identifier, None)
                if key:
                    path = self.embedding_root / f"{key}.safetensors"
                    self.keys[identifier] = key
                    self.cache_paths[identifier] = path
                    self.states[identifier] = "ready" if cache_file_is_valid(path) else "missing"
                else:
                    self.states[identifier] = "missing"
                self.encode_futures.pop(identifier, None)
            self.tree, self.records = tree, records
            set_cache_states(tree, self.states)
        self.fingerprints.remove_missing({record.relative_path for record in records.values()})
        return tree

    def manifest(self) -> dict:
        with self.lock:
            set_cache_states(self.tree, self.states)
            return self.tree

    def status(self) -> dict:
        with self.lock:
            counts = {state: 0 for state in ("missing", "queued", "encoding", "ready")}
            for state in self.states.values():
                counts[state] = counts.get(state, 0) + 1
            return {
                "ready": self.ready,
                "error": self.error,
                "device": self.device,
                "cache": {**counts, "total": len(self.records), "gpuResident": len(self.gpu_cache)},
                "queueDepth": self.executor.queue_depth,
                "currentJob": self.executor.current_job,
                "backgroundPaused": self.executor.background_paused,
                "activeFolder": self._active_folder_status(),
                "lastEncodeMs": self.last_encode_ms,
                "lastDecodeMs": self.last_decode_ms,
            }

    def get_record(self, identifier: str) -> ImageRecord:
        try:
            return self.records[identifier]
        except KeyError as error:
            raise KeyError("Unknown image ID.") from error

    def prioritize_folder(self, folder_path: str, selected_id: str | None = None) -> dict:
        folder = [item.id for item in self.records.values() if item.folder_path == folder_path]
        if not folder and folder_path not in self._folder_paths():
            raise KeyError("Unknown folder path.")
        if selected_id is not None and selected_id not in folder:
            raise KeyError("Selected image is not in the requested folder.")
        previous = self.active_folder
        self.active_folder = folder_path
        self.visited_folders.add(folder_path)
        if previous != folder_path:
            for pending_id, future in list(self.preprocess_futures.items()):
                record = self.records.get(pending_id)
                if record and record.folder_path != folder_path and future.cancel():
                    self.preprocess_futures.pop(pending_id, None)
        if previous != folder_path:
            for item in self.records.values():
                if item.folder_path == previous and self.states.get(item.id) in {"missing", "queued"}:
                    self.queue_embedding(item.id, 80)
        selected = selected_id or (folder[0] if folder else None)
        self.selected_image = selected
        self.promote_ids = set()
        if selected:
            index = folder.index(selected)
            self.promote_ids.add(selected)
            self.queue_embedding(selected, 10)
            for neighbor in (index - 1, index + 1):
                if 0 <= neighbor < len(folder):
                    self.promote_ids.add(folder[neighbor])
                    self.queue_embedding(folder[neighbor], 20)
        remainder = [item for item in folder if item not in self.promote_ids]
        self.folder_order = ([selected] if selected else []) + [item for item in folder if item in self.promote_ids and item != selected] + remainder
        for item in remainder:
            self.queue_embedding(item, 30)
        self._prime_preprocessing()
        return self._active_folder_status()

    def prioritize(self, identifier: str) -> concurrent.futures.Future:
        record = self.get_record(identifier)
        self.prioritize_folder(record.folder_path, identifier)
        return self.queue_embedding(identifier, 10)

    def set_interactive(self, active: bool) -> None:
        self.executor.set_interactive(active)

    def _folder_paths(self) -> set[str]:
        paths = {""}
        def visit(folder: dict) -> None:
            paths.add(folder["path"])
            for child in folder["folders"]: visit(child)
        visit(self.tree)
        return paths

    def _active_folder_status(self) -> dict:
        identifiers = [item.id for item in self.records.values() if item.folder_path == self.active_folder]
        return {"path": self.active_folder, "ready": sum(self.states.get(item) == "ready" for item in identifiers), "total": len(identifiers)}

    def _prime_preprocessing(self) -> None:
        for identifier in self.folder_order:
            if len(self.preprocess_futures) >= 2:
                break
            if self.states.get(identifier) not in {"missing", "queued"} or identifier in self.preprocess_futures:
                continue
            record = self.records.get(identifier)
            if record is not None:
                self.preprocess_futures[identifier] = self.preprocess_pool.submit(preprocess_image, record.absolute_path)

    def queue_embedding(self, identifier: str, priority: int):
        with self.lock:
            key = self.keys.get(identifier)
            resident = self.gpu_cache.get(key) if key else None
            if resident is not None:
                completed: concurrent.futures.Future = concurrent.futures.Future()
                completed.set_result(resident)
                return completed
            if self.states.get(identifier) == "ready" and priority >= 30:
                completed: concurrent.futures.Future = concurrent.futures.Future()
                completed.set_result(None)
                return completed
            existing = self.encode_futures.get(identifier)
            if existing and not existing.done():
                self.executor.submit(priority, lambda: self._complete_shared(identifier, existing), key=f"embedding:{identifier}")
                return existing
            if self.states.get(identifier) != "ready":
                self.states[identifier] = "queued"
            shared: concurrent.futures.Future = concurrent.futures.Future()
            self.encode_futures[identifier] = shared
            self.executor.submit(priority, lambda: self._complete_shared(identifier, shared), key=f"embedding:{identifier}")
            return shared

    def _complete_shared(self, identifier: str, shared):
        if shared.done():
            return shared.result()
        try:
            value = self._ensure_embedding(identifier)
            shared.set_result(value)
            return value
        except BaseException as error:
            if identifier in self.records:
                self.states[identifier] = "missing"
            if not shared.done():
                shared.set_exception(error)
            raise

    def _ensure_embedding(self, identifier: str) -> GpuEmbedding | None:
        record = self.records[identifier]
        key = self.keys.get(identifier)
        if not key:
            key = self.fingerprints.resolve_key(record.relative_path, record.absolute_path, self.model_fingerprint)
            self.keys[identifier] = key
            self.cache_paths[identifier] = self.embedding_root / f"{key}.safetensors"
        cached = self.gpu_cache.get(key)
        if cached:
            return cached
        pending = self.pending_arrays.get(identifier)
        if pending is not None:
            arrays, original, reshaped = pending
            if identifier not in self.promote_ids:
                return None
            embedding = self._to_gpu(arrays, original, reshaped)
            self.gpu_cache.put(key, embedding, embedding.size_bytes)
            return embedding
        path = self.cache_paths[identifier]
        if cache_file_is_valid(path):
            try:
                arrays, original, reshaped = load_embedding(path)
                self.states[identifier] = "ready"
                if identifier not in self.promote_ids:
                    return None
                embedding = self._to_gpu(arrays, original, reshaped)
                self.gpu_cache.put(key, embedding, embedding.size_bytes)
                return embedding
            except Exception:
                path.unlink(missing_ok=True)

        self.states[identifier] = "encoding"
        started = time.perf_counter()
        prepared = self.preprocess_futures.pop(identifier, None)
        self._prime_preprocessing()
        pixels, original, reshaped = prepared.result() if prepared else preprocess_image(self.records[identifier].absolute_path)
        assert self.encoder is not None
        values = self.encoder.run(None, {"pixel_values": pixels})
        arrays = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in zip(EMBEDDING_KEYS, values, strict=True)
        }
        self.write_slots.acquire()
        self.pending_arrays[identifier] = (arrays, original, reshaped)
        self.cache_writer.submit(self._persist_embedding, identifier, path, arrays, original, reshaped)
        embedding = self._to_gpu(arrays, original, reshaped) if identifier in self.promote_ids else None
        if embedding is not None:
            self.gpu_cache.put(key, embedding, embedding.size_bytes)
        self.last_encode_ms = (time.perf_counter() - started) * 1000
        self.states[identifier] = "ready"
        return embedding

    def _persist_embedding(self, identifier: str, path: Path, arrays, original, reshaped) -> None:
        try:
            save_embedding(path, arrays, original, reshaped)
            self.states[identifier] = "ready"
        except Exception:
            self.states[identifier] = "missing"
            path.unlink(missing_ok=True)
        finally:
            current = self.pending_arrays.get(identifier)
            if current is not None and current[0] is arrays:
                self.pending_arrays.pop(identifier, None)
            self.write_slots.release()

    def _to_gpu(self, arrays, original, reshaped) -> GpuEmbedding:
        tensors = {
            key: ort.OrtValue.ortvalue_from_numpy(value, "cuda", 0)
            for key, value in arrays.items()
        }
        return GpuEmbedding(
            tensors=tensors,
            original_size=original,
            reshaped_size=reshaped,
            size_bytes=sum(value.nbytes for value in arrays.values()),
        )

    async def prepare(self, identifier: str) -> dict:
        started = time.perf_counter()
        record = self.get_record(identifier)
        with Image.open(record.absolute_path) as image:
            width, height = image.size
        cache_hit = self.states.get(identifier) == "ready"
        future = self.prioritize(identifier)
        await asyncio.wrap_future(future)
        return {
            "imageId": identifier,
            "width": width,
            "height": height,
            "cacheState": "ready",
            "cacheHit": cache_hit,
            "prepareMs": (time.perf_counter() - started) * 1000,
            "encodeMs": 0 if cache_hit else self.last_encode_ms,
        }

    async def segment(self, identifier: str, points: list[dict]) -> tuple[bytes, dict]:
        self.get_record(identifier)
        future = self.executor.submit(0, lambda: self._segment_sync(identifier, points))
        return await asyncio.wrap_future(future)

    def _segment_sync(self, identifier: str, points: list[dict]) -> tuple[bytes, dict]:
        started = time.perf_counter()
        self.promote_ids.add(identifier)
        embedding = self._ensure_embedding(identifier)
        assert embedding is not None
        coordinates = np.asarray(
            [
                [
                    [
                        [point["x"] * embedding.reshaped_size[1], point["y"] * embedding.reshaped_size[0]]
                        for point in points
                    ]
                ]
            ],
            dtype=np.float32,
        )
        labels = np.asarray([[[point["label"] for point in points]]], dtype=np.int64)
        boxes = np.empty((1, 0, 4), dtype=np.float32)
        assert self.decoder is not None
        binding = self.decoder.io_binding()
        binding.bind_cpu_input("input_points", coordinates)
        binding.bind_cpu_input("input_labels", labels)
        binding.bind_cpu_input("input_boxes", boxes)
        for key, tensor in embedding.tensors.items():
            binding.bind_ortvalue_input(key, tensor)
        for output in ("iou_scores", "pred_masks", "object_score_logits"):
            binding.bind_output(output, "cpu")
        self.decoder.run_with_iobinding(binding)
        scores, masks, _ = binding.copy_outputs_to_cpu()
        candidate = int(np.argmax(scores.reshape(-1)))
        selected = masks[0, 0, candidate]
        resized = torch.nn.functional.interpolate(
            torch.from_numpy(selected)[None, None],
            size=embedding.original_size,
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
        mask = resized > MASK_THRESHOLD
        packed = np.packbits(mask.reshape(-1), bitorder="little").tobytes()
        decode_ms = (time.perf_counter() - started) * 1000
        self.last_decode_ms = decode_ms
        return packed, {
            "width": embedding.original_size[1],
            "height": embedding.original_size[0],
            "decodeMs": decode_ms,
            "cacheState": "ready",
        }

    def shutdown(self) -> None:
        self.executor.stop()
        self.preprocess_pool.shutdown(wait=False, cancel_futures=True)
        self.cache_writer.shutdown(wait=True, cancel_futures=False)
        self.fingerprints.flush()


def preprocess_image(path: Path):
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        resized = rgb.resize((INPUT_SIZE[1], INPUT_SIZE[0]), Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 127.5 - 1.0
    pixels = np.ascontiguousarray(array.transpose(2, 0, 1)[None])
    return pixels, (height, width), INPUT_SIZE
