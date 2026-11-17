import threading
from pathlib import Path

import numpy as np

from server.engine import GpuEmbedding, PriorityGpuExecutor, SamEngine
from server.manifest import ImageRecord


def test_gpu_executor_prioritizes_decode_before_background_work():
    executor = PriorityGpuExecutor()
    started = threading.Event()
    release = threading.Event()
    order: list[str] = []

    def blocker():
        order.append("active")
        started.set()
        release.wait(timeout=2)

    first = executor.submit(50, blocker)
    assert started.wait(timeout=2)
    background = executor.submit(100, lambda: order.append("background"))
    selected = executor.submit(10, lambda: order.append("selected"))
    decode = executor.submit(0, lambda: order.append("decode"))
    release.set()
    for future in (first, decode, selected, background):
        future.result(timeout=2)
    executor.stop()
    assert order == ["active", "decode", "selected", "background"]


def test_gpu_executor_reprioritizes_a_key_without_duplicate_work():
    executor = PriorityGpuExecutor()
    started = threading.Event()
    release = threading.Event()
    order: list[str] = []

    def blocker():
        started.set()
        release.wait(timeout=2)

    active = executor.submit(0, blocker)
    assert started.wait(timeout=2)
    original = executor.submit(80, lambda: order.append("old"), key="embedding:a")
    promoted = executor.submit(10, lambda: order.append("new"), key="embedding:a")
    assert original is promoted
    assert executor.queue_depth == 1
    release.set()
    active.result(timeout=2)
    promoted.result(timeout=2)
    executor.stop()
    assert order == ["new"]


def test_interactive_mode_pauses_background_but_allows_decode():
    executor = PriorityGpuExecutor()
    order: list[str] = []
    executor.set_interactive(True)
    background = executor.submit(30, lambda: order.append("background"), key="embedding:a")
    decode = executor.submit(0, lambda: order.append("decode"))
    decode.result(timeout=2)
    assert order == ["decode"]
    executor.set_interactive(False)
    background.result(timeout=2)
    executor.stop()
    assert order == ["decode", "background"]


def test_folder_priority_orders_selected_neighbors_and_demotes_previous(tmp_path: Path):
    engine = SamEngine(tmp_path)
    records = {}
    for folder, names in (("train", ["a", "b", "c", "d"]), ("val", ["e"])):
        for name in names:
            path = tmp_path / "data" / folder / f"{name}.png"
            records[name] = ImageRecord(name, path.name, f"{folder}/{path.name}", path, folder)
    engine.records = records
    engine.states = {name: "missing" for name in records}
    engine.tree = {"name": "Data", "path": "", "images": [], "folders": [
        {"name": "train", "path": "train", "images": [], "folders": []},
        {"name": "val", "path": "val", "images": [], "folders": []},
    ]}
    calls: list[tuple[str, int]] = []
    engine.queue_embedding = lambda identifier, priority: calls.append((identifier, priority))
    engine._prime_preprocessing = lambda: None
    engine.prioritize_folder("train", "b")
    assert calls[:4] == [("b", 10), ("a", 20), ("c", 20), ("d", 30)]
    calls.clear()
    engine.prioritize_folder("val", "e")
    assert calls[-1] == ("e", 10)
    assert {(name, priority) for name, priority in calls[:-1]} == {
        ("a", 80), ("b", 80), ("c", 80), ("d", 80)}
    engine.shutdown()


def test_session_interactions_are_independent(tmp_path: Path):
    engine = SamEngine(tmp_path)
    assert engine.set_interactive(True, 'alice')
    assert engine.set_interactive(True, 'bob')
    assert engine.set_interactive(False, 'alice')
    assert engine.executor.background_paused
    assert not engine.remove_session('bob')
    assert not engine.executor.background_paused
    engine.shutdown()


def test_session_folder_promotions_are_shared(tmp_path: Path):
    engine = SamEngine(tmp_path)
    engine.records = {
        name: ImageRecord(name, name, f'{folder}/{name}.png', tmp_path / name, folder)
        for folder, names in [('a', ['a1', 'a2']), ('b', ['b1', 'b2'])]
        for name in names
    }
    engine.states = {name: 'missing' for name in engine.records}
    engine.queue_embedding = lambda identifier, priority: None
    engine._prime_preprocessing = lambda: None
    engine.prioritize_folder('a', 'a1', 'alice')
    engine.prioritize_folder('b', 'b1', 'bob')
    assert engine.promote_ids == {'a1', 'a2', 'b1', 'b2'}
    engine.remove_session('alice')
    assert engine.promote_ids == {'b1', 'b2'}
    engine.shutdown()


def test_segment_does_not_copy_unused_decoder_logits(tmp_path: Path):
    class Binding:
        def __init__(self):
            self.outputs = []

        def bind_cpu_input(self, *_args):
            pass

        def bind_ortvalue_input(self, *_args):
            pass

        def bind_output(self, name, _device):
            self.outputs.append(name)

        def copy_outputs_to_cpu(self):
            scores = np.asarray([[[0.1, 0.9, 0.2]]], dtype=np.float32)
            masks = np.ones((1, 1, 3, 2, 2), dtype=np.float32)
            return [scores, masks]

    class Decoder:
        def __init__(self):
            self.binding = Binding()

        def io_binding(self):
            return self.binding

        def run_with_iobinding(self, _binding):
            pass

    engine = SamEngine(tmp_path)
    decoder = Decoder()
    engine.decoder = decoder
    engine._ensure_embedding = lambda _identifier: GpuEmbedding({}, (4, 4), (1008, 1008), 0)
    try:
        packed, metadata = engine._segment_sync("image", [{"x": 0.5, "y": 0.5, "label": 1}])
        assert decoder.binding.outputs == ["iou_scores", "pred_masks"]
        assert len(packed) == 2
        assert metadata["width"] == metadata["height"] == 4
    finally:
        engine.shutdown()
