import threading

from pathlib import Path

from server.engine import PriorityGpuExecutor, SamEngine
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
