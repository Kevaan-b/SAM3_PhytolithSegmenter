from pathlib import Path

import numpy as np

from server.cache import (
    EMBEDDING_KEYS,
    EXPECTED_SHAPES,
    ByteLru,
    cache_file_is_valid,
    cache_key,
    load_embedding,
    save_embedding,
)
from server.manifest import image_id, scan_data


def test_embedding_round_trip(tmp_path: Path):
    arrays = {
        key: np.zeros(shape, dtype=np.float32)
        for key, shape in zip(EMBEDDING_KEYS, EXPECTED_SHAPES, strict=True)
    }
    target = tmp_path / "embedding.safetensors"
    save_embedding(target, arrays, (512, 640), (1008, 1008))
    loaded, original, reshaped = load_embedding(target)
    assert cache_file_is_valid(target)
    assert original == (512, 640)
    assert reshaped == (1008, 1008)
    assert [loaded[key].shape for key in EMBEDDING_KEYS] == list(EXPECTED_SHAPES)


def test_byte_lru_evicts_oldest():
    cache = ByteLru[str](10)
    cache.put("a", "first", 6)
    cache.put("b", "second", 6)
    assert cache.get("a") is None
    assert cache.get("b") == "second"


def test_cache_key_changes_with_image_and_model(tmp_path: Path):
    image = tmp_path / "image.png"
    image.write_bytes(b"first")
    initial = cache_key(image, "model-a")
    image.write_bytes(b"second")
    assert cache_key(image, "model-a") != initial
    assert cache_key(image, "model-b") != cache_key(image, "model-a")


def test_manifest_is_recursive_and_ignores_symlinks(tmp_path: Path):
    data = tmp_path / "data"
    nested = data / "train" / "nested"
    nested.mkdir(parents=True)
    image = nested / "sample.png"
    image.write_bytes(b"image")
    (nested / "ignored.txt").write_text("no")
    (data / "metadata").mkdir()
    (data / "annotations").mkdir()
    (data / "escape.png").symlink_to(image)
    tree, records = scan_data(data)
    found = tree["folders"][0]["folders"][0]["images"][0]
    assert found["id"] == image_id("train/nested/sample.png")
    assert found["url"] == "/data/train/nested/sample.png"
    assert len(records) == 1
    assert [folder["name"] for folder in tree["folders"]] == ["train"]
