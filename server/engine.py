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
    cache_key,
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
    future: concurrent.futures.Future = field(compare=False)
    callback: Callable = field(compare=False)


class PriorityGpuExecutor:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._jobs: list[QueuedJob] = []
        self._sequence = 0
        self._stopped = False
        self._thread = threading.Thread(target=self._run, daemon=True, name="sam3-gpu")
        self._thread.start()

    def submit(self, priority: int, callback: Callable):
        future: concurrent.futures.Future = concurrent.futures.Future()
        with self._condition:
            self._sequence += 1
            heapq.heappush(
                self._jobs,
                QueuedJob(priority, self._sequence, future, callback),
            )
            self._condition.notify()
        return future

    @property
    def queue_depth(self) -> int:
        with self._condition:
            return len(self._jobs)

    def stop(self) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._jobs and not self._stopped:
                    self._condition.wait()
                if self._stopped:
                    return
                job = heapq.heappop(self._jobs)
            if job.future.done():
                continue
            try:
                job.future.set_result(job.callback())
            except BaseException as error:
                job.future.set_exception(error)


class SamEngine:
    def __init__(self, project_root: Path, gpu_budget_gib: float = 16) -> None:
        self.project_root = project_root.resolve()
        self.data_root = self.project_root / "data"
        self.cache_root = self.project_root / ".samotator-cache"
        self.embedding_root = self.cache_root / "embeddings"
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
                key = cache_key(record.absolute_path, self.model_fingerprint)
                path = self.embedding_root / f"{key}.safetensors"
                self.keys[identifier] = key
                self.cache_paths[identifier] = path
                self.states[identifier] = "ready" if cache_file_is_valid(path) else "missing"
            active_paths = set(self.cache_paths.values())
            for stale in self.embedding_root.glob("*.safetensors"):
                if stale not in active_paths:
                    stale.unlink(missing_ok=True)

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
            for identifier in self.records:
                self.queue_embedding(identifier, 100)
        except BaseException as error:
            self.error = str(error)
            self.device = "error"

    def refresh_manifest(self) -> dict:
        tree, records = scan_data(self.data_root)
        changed: list[str] = []
        with self.lock:
            removed = set(self.records) - set(records)
            for identifier in removed:
                self.states.pop(identifier, None)
                self.keys.pop(identifier, None)
                self.cache_paths.pop(identifier, None)
                self.encode_futures.pop(identifier, None)
                self.gpu_cache.discard(identifier)
            for identifier, record in records.items():
                if not self.model_fingerprint:
                    self.states.setdefault(identifier, "missing")
                    continue
                key = cache_key(record.absolute_path, self.model_fingerprint)
                path = self.embedding_root / f"{key}.safetensors"
                if self.keys.get(identifier) == key:
                    continue
                self.gpu_cache.discard(identifier)
                self.keys[identifier] = key
                self.cache_paths[identifier] = path
                self.states[identifier] = "ready" if cache_file_is_valid(path) else "missing"
                self.encode_futures.pop(identifier, None)
                changed.append(identifier)
            self.tree, self.records = tree, records
            set_cache_states(tree, self.states)
        if self.ready:
            for identifier in changed:
                self.queue_embedding(identifier, 100)
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
                "lastEncodeMs": self.last_encode_ms,
                "lastDecodeMs": self.last_decode_ms,
            }

    def get_record(self, identifier: str) -> ImageRecord:
        try:
            return self.records[identifier]
        except KeyError as error:
            raise KeyError("Unknown image ID.") from error

    def prioritize(self, identifier: str) -> concurrent.futures.Future:
        record = self.get_record(identifier)
        future = self.queue_embedding(identifier, 10)
        folder = [
            item.id for item in self.records.values() if item.folder_path == record.folder_path
        ]
        index = folder.index(identifier)
        for neighbor in (index - 1, index + 1):
            if 0 <= neighbor < len(folder):
                self.queue_embedding(folder[neighbor], 20)
        for item in folder:
            if item != identifier:
                self.queue_embedding(item, 30)
        return future

    def queue_embedding(self, identifier: str, priority: int):
        with self.lock:
            resident = self.gpu_cache.get(identifier)
            if resident is not None:
                completed: concurrent.futures.Future = concurrent.futures.Future()
                completed.set_result(resident)
                return completed
            existing = self.encode_futures.get(identifier)
            if existing and not existing.done():
                # Add a higher-priority route to the same computation.
                if priority < 100:
                    self.executor.submit(priority, lambda: self._complete_shared(identifier, existing))
                return existing
            if self.states.get(identifier) != "ready":
                self.states[identifier] = "queued"
            shared: concurrent.futures.Future = concurrent.futures.Future()
            self.encode_futures[identifier] = shared
            self.executor.submit(priority, lambda: self._complete_shared(identifier, shared))
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

    def _ensure_embedding(self, identifier: str) -> GpuEmbedding:
        cached = self.gpu_cache.get(identifier)
        if cached:
            return cached
        path = self.cache_paths[identifier]
        if cache_file_is_valid(path):
            try:
                arrays, original, reshaped = load_embedding(path)
                embedding = self._to_gpu(arrays, original, reshaped)
                self.gpu_cache.put(identifier, embedding, embedding.size_bytes)
                self.states[identifier] = "ready"
                return embedding
            except Exception:
                path.unlink(missing_ok=True)

        self.states[identifier] = "encoding"
        started = time.perf_counter()
        pixels, original, reshaped = preprocess_image(self.records[identifier].absolute_path)
        assert self.encoder is not None
        values = self.encoder.run(None, {"pixel_values": pixels})
        arrays = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in zip(EMBEDDING_KEYS, values, strict=True)
        }
        save_embedding(path, arrays, original, reshaped)
        embedding = self._to_gpu(arrays, original, reshaped)
        self.gpu_cache.put(identifier, embedding, embedding.size_bytes)
        self.last_encode_ms = (time.perf_counter() - started) * 1000
        self.states[identifier] = "ready"
        return embedding

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
        embedding = self._ensure_embedding(identifier)
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


def preprocess_image(path: Path):
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        resized = rgb.resize((INPUT_SIZE[1], INPUT_SIZE[0]), Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 127.5 - 1.0
    pixels = np.ascontiguousarray(array.transpose(2, 0, 1)[None])
    return pixels, (height, width), INPUT_SIZE
