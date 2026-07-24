# Recommended COCO Dataset Format

This document defines the recommended dataset structure for multi-object phytolith detection, classification, and instance segmentation.

The canonical dataset format should be **COCO instance segmentation** with:

- one annotation per object instance
- one category ID per class
- an instance mask for every object
- a bounding box derived from the mask
- mask area derived from the mask
- optional image-level and object-level metadata

This single master format can support:

- bounding-box detection
- instance segmentation
- class-specific prediction filtering
- object crop classification
- semantic-mask generation
- DINOv3-backed detection and segmentation models

---

## 1. Recommended Folder Structure

```text
phytolith_dataset/
├── images/
│   ├── train/
│   │   ├── image_000001.png
│   │   ├── image_000002.png
│   │   └── ...
│   ├── val/
│   │   ├── image_000101.png
│   │   └── ...
│   └── test/
│       ├── image_000201.png
│       └── ...
│
├── annotations/
│   ├── instances_train.json
│   ├── instances_val.json
│   └── instances_test.json
│
├── metadata/
│   ├── categories.json
│   └── samples.csv
│
└── README.md
```

### Folder responsibilities

| Path | Purpose |
|---|---|
| `images/train/` | Training images |
| `images/val/` | Validation images |
| `images/test/` | Test images |
| `annotations/instances_train.json` | COCO annotations for training images |
| `annotations/instances_val.json` | COCO annotations for validation images |
| `annotations/instances_test.json` | COCO annotations for test images |
| `metadata/categories.json` | Optional readable category mapping |
| `metadata/samples.csv` | Optional image, sample, slide, source, and microscope metadata |

Use the same class-to-ID mapping in every split.

---

## 2. Canonical COCO Structure

Each COCO annotation file contains five top-level fields:

```json
{
  "info": {},
  "licenses": [],
  "images": [],
  "annotations": [],
  "categories": []
}
```

The important sections are:

- `images`: one record per image
- `annotations`: one record per object instance
- `categories`: one record per class

---

## 3. Complete Example

The following example contains one image with four object instances:

- two bilobates
- one rondel
- one cross

The two bilobates share the same class but have different masks and annotation IDs. Masks may overlap.

```json
{
  "info": {
    "description": "Phytolith instance segmentation dataset",
    "version": "1.0",
    "year": 2026
  },

  "licenses": [],

  "categories": [
    {
      "id": 1,
      "name": "bilobate",
      "supercategory": "phytolith"
    },
    {
      "id": 2,
      "name": "rondel",
      "supercategory": "phytolith"
    },
    {
      "id": 3,
      "name": "cross",
      "supercategory": "phytolith"
    }
  ],

  "images": [
    {
      "id": 1,
      "file_name": "train/image_000001.png",
      "width": 2048,
      "height": 1536,

      "source": "source_a",
      "sample_id": "sample_001",
      "slide_id": "slide_003",
      "magnification": 400
    }
  ],

  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [120, 180, 95, 73],
      "segmentation": {
        "size": [1536, 2048],
        "counts": "encoded_rle_mask_1"
      },
      "area": 4812,
      "iscrowd": 0,
      "attributes": {
        "occluded": false,
        "truncated": false,
        "fragmented": false,
        "ambiguous": false,
        "confidence": "high"
      }
    },
    {
      "id": 2,
      "image_id": 1,
      "category_id": 1,
      "bbox": [340, 225, 84, 102],
      "segmentation": {
        "size": [1536, 2048],
        "counts": "encoded_rle_mask_2"
      },
      "area": 5231,
      "iscrowd": 0,
      "attributes": {
        "occluded": true,
        "truncated": false,
        "fragmented": false,
        "ambiguous": false,
        "confidence": "medium"
      }
    },
    {
      "id": 3,
      "image_id": 1,
      "category_id": 2,
      "bbox": [620, 410, 110, 96],
      "segmentation": {
        "size": [1536, 2048],
        "counts": "encoded_rle_mask_3"
      },
      "area": 6780,
      "iscrowd": 0,
      "attributes": {
        "occluded": false,
        "truncated": false,
        "fragmented": false,
        "ambiguous": false,
        "confidence": "high"
      }
    },
    {
      "id": 4,
      "image_id": 1,
      "category_id": 3,
      "bbox": [675, 445, 125, 108],
      "segmentation": {
        "size": [1536, 2048],
        "counts": "encoded_rle_mask_4"
      },
      "area": 7345,
      "iscrowd": 0,
      "attributes": {
        "occluded": true,
        "truncated": false,
        "fragmented": false,
        "ambiguous": true,
        "confidence": "medium"
      }
    }
  ]
}
```

---

## 4. Image Records

Each image requires:

```json
{
  "id": 1,
  "file_name": "train/image_000001.png",
  "width": 2048,
  "height": 1536
}
```

### Required fields

| Field | Meaning |
|---|---|
| `id` | Unique integer image ID |
| `file_name` | Image path relative to the dataset image root |
| `width` | Image width in pixels |
| `height` | Image height in pixels |

### Recommended optional metadata

```json
{
  "source": "source_a",
  "sample_id": "sample_001",
  "slide_id": "slide_003",
  "specimen_id": "specimen_012",
  "microscope": "scope_01",
  "magnification": 400
}
```

These are custom fields rather than core COCO fields. Most loaders will ignore unknown fields safely.

For larger metadata tables, store the information separately in `metadata/samples.csv`.

Example:

```csv
image_id,file_name,source,sample_id,slide_id,specimen_id,microscope,magnification
1,train/image_000001.png,source_a,sample_001,slide_003,specimen_012,scope_01,400
```

---

## 5. Category Records

Each class receives one stable integer ID.

```json
{
  "id": 1,
  "name": "bilobate",
  "supercategory": "phytolith"
}
```

### Recommended rules

- start class IDs at `1`
- reserve `0` for background internally
- never reuse an ID for a different class
- use the same IDs in train, validation, and test files
- use lowercase, consistent class names

Example `metadata/categories.json`:

```json
{
  "1": "bilobate",
  "2": "rondel",
  "3": "cross"
}
```

---

## 6. Annotation Records

Every visible object instance receives its own annotation record.

```json
{
  "id": 17,
  "image_id": 4,
  "category_id": 2,
  "bbox": [120, 85, 64, 91],
  "segmentation": {
    "size": [768, 1024],
    "counts": "encoded_rle_data"
  },
  "area": 4210,
  "iscrowd": 0
}
```

### Required fields

| Field | Meaning |
|---|---|
| `id` | Unique annotation ID |
| `image_id` | ID of the image containing the object |
| `category_id` | Class ID of the object |
| `bbox` | Object bounding box in COCO XYWH format |
| `segmentation` | Polygon or RLE instance mask |
| `area` | Number of pixels inside the instance mask |
| `iscrowd` | Usually `0` for individually annotated objects |

---

## 7. One Annotation per Object Instance

Two objects from the same class must remain separate.

```text
Annotation 1 -> bilobate instance A
Annotation 2 -> bilobate instance B
Annotation 3 -> rondel instance A
Annotation 4 -> cross instance A
```

The two bilobates share:

```json
"category_id": 1
```

but have different:

- annotation IDs
- masks
- bounding boxes
- areas

Do not merge same-class objects into a single mask when instance detection is required.

---

## 8. Bounding-Box Format

COCO stores bounding boxes as:

```text
[x_min, y_min, width, height]
```

Example:

```json
"bbox": [120, 180, 95, 73]
```

This means:

```text
left   = 120
top    = 180
width  = 95
height = 73
```

It does not mean:

```text
[x_min, y_min, x_max, y_max]
```

### Recommended generation rule

Always derive the box from the binary instance mask:

```text
x_min = minimum x-coordinate containing foreground
x_max = maximum x-coordinate containing foreground
y_min = minimum y-coordinate containing foreground
y_max = maximum y-coordinate containing foreground

width  = x_max - x_min + 1
height = y_max - y_min + 1
```

The `+1` depends on whether the implementation treats pixel coordinates as inclusive. Use one convention consistently and validate against the target framework.

---

## 9. Segmentation Format

COCO supports polygon and run-length encoded masks.

### 9.1 Polygon format

```json
{
  "segmentation": [
    [
      120, 190,
      135, 180,
      170, 182,
      205, 210,
      212, 240,
      180, 252,
      145, 245,
      122, 220
    ]
  ]
}
```

Each coordinate pair is:

```text
x, y
```

An object may contain multiple polygons:

```json
{
  "segmentation": [
    [10, 10, 30, 10, 30, 30, 10, 30],
    [50, 50, 60, 50, 60, 60, 50, 60]
  ]
}
```

Polygon masks work well for simple, continuous boundaries.

### 9.2 RLE format

```json
{
  "segmentation": {
    "size": [1536, 2048],
    "counts": "encoded_rle_string"
  }
}
```

The `size` order is:

```text
[image_height, image_width]
```

### Recommended choice

Use **RLE masks** as the canonical export for phytoliths because they preserve:

- irregular boundaries
- holes
- narrow structures
- disconnected regions
- detailed pixel-level masks
- overlapping instances

Polygon export can remain an optional compatibility mode.

---

## 10. Area Calculation

For binary masks:

```python
area = int(mask.sum())
```

`area` should represent the number of foreground pixels in the instance mask.

Do not calculate area as:

```text
bbox_width * bbox_height
```

unless the mask completely fills the box.

---

## 11. Overlapping Masks

Overlapping instance masks are valid.

Each object retains an independent mask:

```text
object A mask -> annotation A
object B mask -> annotation B
```

A pixel may therefore belong to more than one annotation.

Do not flatten overlapping instance masks into one master class map. Flattening loses:

- instance identity
- overlap information
- same-class instance separation
- object-specific attributes

### Visible versus amodal masks

Use a consistent annotation policy:

- **visible mask**: annotate only the visible object region
- **amodal mask**: estimate the complete object, including hidden regions

Standard COCO-style instance segmentation normally uses visible masks.

Record occlusion explicitly when useful:

```json
{
  "attributes": {
    "occluded": true
  }
}
```

---

## 12. Recommended Object Attributes

Optional object-level attributes may be stored under a custom `attributes` field.

```json
{
  "attributes": {
    "occluded": false,
    "truncated": false,
    "fragmented": false,
    "ambiguous": false,
    "confidence": "high",
    "annotator": "annotator_01"
  }
}
```

Recommended meanings:

| Attribute | Meaning |
|---|---|
| `occluded` | Part of the object is blocked by another object |
| `truncated` | Object extends beyond the image boundary |
| `fragmented` | Object is physically incomplete or broken |
| `ambiguous` | Class label or boundary is uncertain |
| `confidence` | Annotation confidence, such as `low`, `medium`, or `high` |
| `annotator` | Optional annotator identifier |

These fields are not part of the strict COCO core. Training code may ignore them.

---

## 13. Class-Specific Detection

No extra annotation structure is required to detect or filter a particular class.

The class is defined by:

```json
"category_id": 2
```

A trained detector may return:

```python
prediction = {
    "boxes": boxes,
    "labels": labels,
    "scores": scores,
    "masks": masks
}
```

Filtering to one class is performed at inference time:

```python
target_class_id = 2
keep = prediction["labels"] == target_class_id

class_boxes = prediction["boxes"][keep]
class_scores = prediction["scores"][keep]
class_masks = prediction["masks"][keep]
```

This is closed-set class filtering. The requested class must exist in the training categories.

Arbitrary text-prompted detection requires a text-conditioned or open-vocabulary model. It is a model architecture decision, not a COCO formatting decision.

---

## 14. Supported Training Tasks

### Bounding-box detection

Use:

- `bbox`
- `category_id`

The segmentation field can be ignored.

### Instance segmentation

Use:

- `bbox`
- `segmentation`
- `category_id`
- `area`

### Object classification

Generate one crop per annotation using its bounding box or mask.

Recommended crop labels come from:

```json
"category_id"
```

### Semantic segmentation

Generate a class-ID image from the instance masks.

Example class map:

```text
0 = background
1 = bilobate
2 = rondel
3 = cross
```

For overlapping masks, define a deterministic conversion policy because one semantic output pixel normally stores only one class.

### DINOv3-backed models

DINOv3 is typically used as the visual backbone. The COCO annotations are consumed by a detection or segmentation head such as:

- Faster R-CNN
- Mask R-CNN
- Cascade R-CNN
- Deformable DETR
- DINO detector

DINOv3 itself does not change the annotation format.

---

## 15. Recommended Internal Annotation Representation

The annotation application may use a richer internal structure before exporting COCO.

```json
{
  "instance_id": "instance_000124",
  "category_id": 2,
  "category_name": "rondel",
  "mask": "binary_mask_or_rle",
  "bbox_xywh": [620, 410, 110, 96],
  "area": 6780,
  "attributes": {
    "occluded": false,
    "fragmented": false,
    "ambiguous": false
  }
}
```

Recommended source of truth:

```text
binary instance mask
```

Derive the following from the mask during export:

- RLE segmentation
- bounding box
- area

This reduces mismatches between masks, boxes, and areas.

---

## 16. Validation Rules

Before training, validate that:

- every image ID is unique
- every annotation ID is unique
- every `image_id` points to an existing image
- every `category_id` points to an existing category
- category IDs are identical across dataset splits
- image dimensions match the actual files
- mask dimensions match image dimensions
- RLE `size` uses `[height, width]`
- bounding boxes use `[x, y, width, height]`
- bounding boxes remain inside image boundaries
- area is positive for valid instances
- boxes and areas match their masks
- empty images are handled intentionally
- annotation JSON parses without errors

Images without objects may remain in the dataset with no matching annotation entries. Confirm that the selected training framework supports empty targets.

---

## 17. Final Recommendation

Use the following as the canonical dataset standard:

```text
COCO instance segmentation
+ one annotation per object instance
+ RLE masks
+ automatically derived bounding boxes
+ automatically derived mask areas
+ stable category IDs
+ optional image and object metadata
```

The binary instance mask should be treated as the primary annotation. Bounding boxes and areas should be regenerated from the mask whenever annotations are edited.

This structure preserves the maximum useful information while remaining compatible with standard object detection and instance segmentation workflows.
