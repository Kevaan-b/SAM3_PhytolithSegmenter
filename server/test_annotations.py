import base64
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask

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

    output = json.loads((tmp_path / "data" / "annotations" / "instances_train.json").read_text())
    assert output["images"] == [{"id": 1, "file_name": "train/sample.png", "width": 5, "height": 4}]
    annotation = output["annotations"][0]
    assert annotation["category_id"] == 1
    assert annotation["area"] == 3
    assert annotation["bbox"] == [1.0, 1.0, 2.0, 2.0]
    encoded = {"size": annotation["segmentation"]["size"], "counts": annotation["segmentation"]["counts"].encode("ascii")}
    decoded = coco_mask.decode(encoded)
    assert np.flatnonzero(decoded.reshape(-1)).tolist() == [6, 7, 12]


def test_empty_layers_are_drafts_but_not_coco_annotations(tmp_path: Path):
    store, _ = fixture(tmp_path)
    empty = packed(5, 4, [])
    result = store.save_image(
        "image-a", 5, 4,
        [{"layerId": "empty", "categoryId": 1, "rawMask": empty, "effectiveMask": empty}],
        "empty", False,
    )
    assert result == {"imageId": "image-a", "savedLayers": 0, "emptyLayers": 1, "savedAt": result["savedAt"]}
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
