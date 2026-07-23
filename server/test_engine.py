import threading

from server.engine import PriorityGpuExecutor


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
