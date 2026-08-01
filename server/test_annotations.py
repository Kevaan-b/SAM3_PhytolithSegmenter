import base64
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask

from server.annotation_index import AnnotationIndex
from server.annotations import AnnotationStore
from server.manifest import ImageRecord


def fixture(tmp_path: Path):
    data = tmp_path / "data"
    path = data / "train" / "sample.png"
    path.parent.mkdir(parents=True)
    Image.new("RGB", (5, 4)).save(path)
    record = ImageRecord("image-a", "sample.png", "train/sample.png", path, "train")
    store = AnnotationStore(data, lambda identifier: record if identifier == record.id else (_ for _ in ()).throw(KeyError(identifier)))
    return store, record


def packed(width: int, height: int, enabled: list[int]) -> str:
    values = np.zeros(width * height, dtype=np.uint8)
    values[enabled] = 1
    return base64.b64encode(np.packbits(values, bitorder="little").tobytes()).decode("ascii")


def test_categories_are_stable_editable_and_archivable(tmp_path: Path):
    store, _ = fixture(tmp_path)
    initial = store.categories()["categories"][0]
    assert initial == {"id": 1, "name": "object", "supercategory": "phytolith", "color": "#4094dc", "active": True}
    second = store.add_category(" rondel ", color="#AABBCC")
    assert second["id"] == 2
    assert second["color"] == "#aabbcc"
    assert store.update_category(2, {"name": "cross"})["name"] == "cross"
    assert store.archive_category(2)["active"] is False
    assert store.add_category("rondel")["id"] == 3


def test_save_round_trip_and_coco_mask_derivatives(tmp_path: Path):
    store, _ = fixture(tmp_path)
    mask = packed(5, 4, [6, 7, 12])
    result = store.save_image(
        "image-a", 5, 4,
        [{"layerId": "layer-a", "categoryId": 1, "rawMask": mask, "effectiveMask": mask}],
        "layer-a", False,
    )
    assert result["savedLayers"] == 1
    restored = store.load_image("image-a")
    assert restored["layers"][0]["layerId"] == "layer-a"
    assert restored["layers"][0]["rawMask"] == mask

    store.flush_exports()
    output = json.loads((tmp_path / "data" / "annotations" / "instances_train.json").read_text())
    assert output["images"] == [{"id": 1, "file_name": "train/sample.png", "width": 5, "height": 4}]
    annotation = output["annotations"][0]
    assert annotation["category_id"] == 1
    assert annotation["area"] == 3
    assert annotation["bbox"] == [1.0, 1.0, 2.0, 2.0]
    encoded = {"size": annotation["segmentation"]["size"], "counts": annotation["segmentation"]["counts"].encode("ascii")}
    decoded = coco_mask.decode(encoded)
    assert np.flatnonzero(decoded.reshape(-1)).tolist() == [6, 7, 12]


def test_statistics_counts_saved_annotations_by_class(tmp_path: Path):
    store, _ = fixture(tmp_path)
    second = store.add_category("cross")
    mask = packed(5, 4, [0])
    store.save_image(
        "image-a", 5, 4,
        [
            {"layerId": "object", "categoryId": 1, "rawMask": mask, "effectiveMask": mask},
            {"layerId": "cross", "categoryId": second["id"], "rawMask": mask, "effectiveMask": mask},
        ],
        "cross", False,
    )
    statistics = store.statistics()
    assert statistics["totalAnnotations"] == 2
    assert [item["annotationCount"] for item in statistics["classes"]] == [1, 1]
    previews = store.statistics_previews(1)
    assert len(previews) == 1
    assert previews[0]["imageId"] == "image-a"
    assert previews[0]["annotationCount"] == 1
    assert store.preview_image("image-a", 1).startswith(b"RIFF")


def test_empty_layers_are_drafts_but_not_coco_annotations(tmp_path: Path):
    store, _ = fixture(tmp_path)
    empty = packed(5, 4, [])
    result = store.save_image(
        "image-a", 5, 4,
        [{"layerId": "empty", "categoryId": 1, "rawMask": empty, "effectiveMask": empty}],
        "empty", False,
    )
    assert result == {"imageId": "image-a", "savedLayers": 0, "emptyLayers": 1, "savedAt": result["savedAt"]}
    store.flush_exports()
    output = json.loads((tmp_path / "data" / "annotations" / "instances_train.json").read_text())
    assert output["annotations"] == []


def test_save_rejects_bad_dimensions_categories_and_masks(tmp_path: Path):
    import pytest
    store, _ = fixture(tmp_path)
    mask = packed(5, 4, [0])
    with pytest.raises(ValueError, match="dimensions"):
        store.save_image("image-a", 4, 4, [], None, False)
    with pytest.raises(ValueError, match="Unknown category"):
        store.save_image("image-a", 5, 4, [{"layerId": "a", "categoryId": 99, "rawMask": mask, "effectiveMask": mask}], None, False)
    with pytest.raises(ValueError, match="byte length"):
        store.save_image("image-a", 5, 4, [{"layerId": "a", "categoryId": 1, "rawMask": base64.b64encode(b"x").decode(), "effectiveMask": mask}], None, False)


def test_statistics_preview_query_is_capped_at_eight(tmp_path: Path):
    data = tmp_path / "data"
    records: dict[str, ImageRecord] = {}
    for index in range(10):
        path = data / "train" / f"sample-{index}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (5, 4)).save(path)
        records[f"image-{index}"] = ImageRecord(
            f"image-{index}", path.name, f"train/{path.name}", path, "train")
    store = AnnotationStore(data, records.__getitem__)
    mask = packed(5, 4, [0])
    for index in range(10):
        store.save_image(f"image-{index}", 5, 4, [{
            "layerId": f"layer-{index}", "categoryId": 1,
            "rawMask": mask, "effectiveMask": mask,
        }], f"layer-{index}", False)
    assert len(store.statistics_previews(1, 100)) == 8
    assert store.statistics()["totalAnnotations"] == 10
    store.shutdown()


def test_save_path_does_not_scan_unrelated_drafts(tmp_path: Path):
    store, _ = fixture(tmp_path)
    store._drafts = lambda: (_ for _ in ()).throw(AssertionError("full draft scan"))
    mask = packed(5, 4, [0])
    store.save_image("image-a", 5, 4, [{
        "layerId": "layer-a", "categoryId": 1,
        "rawMask": mask, "effectiveMask": mask,
    }], "layer-a", False)
    assert store.statistics()["totalAnnotations"] == 1
    store.shutdown()


def test_corrupt_derived_index_recovers_from_drafts(tmp_path: Path):
    store, _ = fixture(tmp_path)
    mask = packed(5, 4, [0])
    store.save_image("image-a", 5, 4, [{
        "layerId": "layer-a", "categoryId": 1,
        "rawMask": mask, "effectiveMask": mask,
    }], "layer-a", False)
    store.shutdown()
    index_path = tmp_path / ".samotator-cache" / "annotations.sqlite3"
    index_path.write_bytes(b"not a sqlite database")
    recovered = AnnotationIndex(index_path, tmp_path / "data" / ".samotator" / "annotations")
    recovered.reconcile()
    assert recovered.statistics([])["totalAnnotations"] == 1
