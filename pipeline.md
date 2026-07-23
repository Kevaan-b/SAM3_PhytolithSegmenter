# SAM3 Q4 Local Interactive Segmentation Pipeline

## Scope

This document describes how to use the model bundle downloaded to
`sam3-q4/` for local, interactive image segmentation with positive points,
negative points, or a bounding box.

The bundle reproduces the core inference path used by the SegmentLens SAM3
playground:

```text
image
  -> image processor
  -> Q4 vision encoder
  -> three image-embedding tensors
  -> point/box prompt encoder and mask decoder
  -> three candidate masks plus quality scores
  -> resize, threshold, select, and render
```

This is an interactive image-segmentation bundle. Its exported ONNX interface
does not include text-prompt inputs or the full video-memory/tracking pipeline,
even though `config.json` describes a `sam3_tracker` model family.

## Downloaded files and their roles

Keep the files in this relative layout:

```text
sam3-q4/
├── config.json
├── preprocessor_config.json
└── onnx/
    ├── vision_encoder_q4.onnx
    ├── vision_encoder_q4.onnx_data
    ├── prompt_encoder_mask_decoder.onnx
    └── prompt_encoder_mask_decoder.onnx_data
```

| File | Size | Role |
|---|---:|---|
| `config.json` | 5,078 bytes | Model architecture and Transformers.js metadata |
| `preprocessor_config.json` | 702 bytes | RGB conversion, resize, normalization, and mask post-processing settings |
| `vision_encoder_q4.onnx` | 1,320,396 bytes | Q4 vision-encoder graph |
| `vision_encoder_q4.onnx_data` | 368,954,368 bytes | External weights for the Q4 vision encoder |
| `prompt_encoder_mask_decoder.onnx` | 213,114 bytes | Point/box prompt encoder and mask-decoder graph |
| `prompt_encoder_mask_decoder.onnx_data` | 22,072,320 bytes | External weights for the prompt encoder and mask decoder |

The `.onnx` files are graph wrappers, not self-contained models. Their
`.onnx_data` companions are mandatory and must remain beside them with the
same names. Inspection of the graphs confirmed that all 856 external tensors
in the vision encoder and all 108 external tensors in the decoder resolve
exactly into the downloaded sidecar files.

## Recommended runtime

The closest reproduction of the browser playground is:

- `@huggingface/transformers` 3.8.0, matching the inspected playground;
- `Sam3TrackerModel`;
- `AutoProcessor`;
- ONNX Runtime Web through Transformers.js;
- the WebGPU device;
- Q4 for `vision_encoder`;
- FP32 for `prompt_encoder_mask_decoder`.

The playground's Q4 setting is therefore a mixed-precision pipeline:

```js
{
  vision_encoder: "q4",
  prompt_encoder_mask_decoder: "fp32",
}
```

It is not a fully Q4 model. Only the large vision encoder is loaded from the
Q4 export.

Pin version `3.8.0` when reproducing the inspected implementation. Newer
Transformers.js releases may work, but their model APIs and local-loading
behavior should be verified before upgrading.

WebGPU is the most faithful local target. A direct ONNX Runtime Python
implementation is possible, but the caller must reproduce processor behavior,
empty prompt tensors, coordinate conversion, mask resizing, and thresholding
manually. The Q4 encoder also uses the `com.microsoft` ONNX operator domain, so
the selected runtime must support those operators.

## Serving the model locally

A browser cannot reliably load the bundle directly from `file://`. Serve the
application and model directory over HTTP, and preserve the directory layout.
For example, expose the repository root with a static development server so
that these URLs are available:

```text
http://localhost:<port>/sam3-q4/config.json
http://localhost:<port>/sam3-q4/preprocessor_config.json
http://localhost:<port>/sam3-q4/onnx/vision_encoder_q4.onnx
http://localhost:<port>/sam3-q4/onnx/vision_encoder_q4.onnx_data
http://localhost:<port>/sam3-q4/onnx/prompt_encoder_mask_decoder.onnx
http://localhost:<port>/sam3-q4/onnx/prompt_encoder_mask_decoder.onnx_data
```

The server must serve the large `.onnx_data` files without truncation. Byte
range support is helpful for large model assets. If the model and application
use different origins, configure CORS for model requests.

## Stage 1: initialize the processor and model

The following is a representative Transformers.js setup. The exact bundler
syntax may differ, but the model name, local model path, device, and dtype map
are the important parts.

```js
import {
  AutoProcessor,
  RawImage,
  Sam3TrackerModel,
  Tensor,
  env,
} from "@huggingface/transformers";

env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = "/";

const model = await Sam3TrackerModel.from_pretrained("sam3-q4", {
  device: "webgpu",
  dtype: {
    vision_encoder: "q4",
    prompt_encoder_mask_decoder: "fp32",
  },
});

const processor = await AutoProcessor.from_pretrained("sam3-q4");
```

With `env.localModelPath = "/"`, the identifier `sam3-q4` resolves to the
served `/sam3-q4/` directory. Adjust `localModelPath` to match the static
server rather than changing the downloaded filenames.

## Stage 2: preprocess the image

The downloaded `preprocessor_config.json` specifies:

- convert the image to RGB;
- resize to a model input of 1008 × 1008;
- rescale bytes by `1 / 255`;
- normalize each channel with mean `0.5` and standard deviation `0.5`;
- return channels first;
- use a nominal mask size of 288 × 288.

The encoder ultimately requires exactly 1008 × 1008. Coordinate scaling must
still use the processor's returned `reshaped_input_sizes`; do not hardcode
1008 into UI-coordinate conversion.

For a byte-valued RGB channel `x`, normalization is:

```text
normalized = ((x / 255) - 0.5) / 0.5
```

This maps `[0, 255]` approximately to `[-1, 1]`.

Use the processor instead of duplicating this logic when possible:

```js
const image = await RawImage.fromURL(imageUrl);
const imageInputs = await processor(image);
```

`imageInputs` must be retained after encoding. In addition to
`pixel_values`, mask post-processing needs:

- `original_sizes`: source image height and width;
- `reshaped_input_sizes`: height and width after processor resizing.

Do not infer these values from the CSS size of a displayed image.

### Vision-encoder input

| Name | Type | Shape |
|---|---|---|
| `pixel_values` | `float32` | `[batch_size, 3, 1008, 1008]` |

## Stage 3: compute image embeddings once

Image encoding is the expensive stage. Run it once per uploaded image and
cache the result while the user adds or changes prompts:

```js
const imageEmbeddings = await model.get_image_embeddings(imageInputs);
```

The Q4 vision encoder produces three feature levels:

| Name | Type | Shape |
|---|---|---|
| `image_embeddings.0` | `float32` | `[B, 32, 288, 288]` |
| `image_embeddings.1` | `float32` | `[B, 64, 144, 144]` |
| `image_embeddings.2` | `float32` | `[B, 256, 72, 72]` |

The shapes above are the exact inputs expected by the downloaded decoder.
Recompute them only when the source image changes.

## Stage 4: construct prompts

The UI may store points and boxes as normalized coordinates in `[0,1]`, but
the decoder receives coordinates in resized-image pixels.

Let:

```text
resized_height = imageInputs.reshaped_input_sizes[0][0]
resized_width  = imageInputs.reshaped_input_sizes[0][1]
```

Convert a normalized point `(x, y)` with:

```text
model_x = x * resized_width
model_y = y * resized_height
```

Convert a normalized box `(x1, y1, x2, y2)` with:

```text
[x1 * resized_width,
 y1 * resized_height,
 x2 * resized_width,
 y2 * resized_height]
```

Coordinates use `(x, y)` order. Sizes use `(height, width)` order.

### Point labels

The playground uses:

| Label | Meaning |
|---:|---|
| `1` | positive/foreground point; left click |
| `0` | negative/background point; right click |

Labels must be `int64`. In JavaScript, construct them with `BigInt`.
For normal interactive prompts, send only `0` and `1`. Padding and empty-prompt
sentinels are model-wrapper details; let Transformers.js construct them.
Direct ragged-batch implementations must reproduce the selected runtime's
sentinel rules rather than treating `-1` as a universal padding label.

### Point tensor shapes

| Name | Type | Shape |
|---|---|---|
| `input_points` | `float32` | `[B, 1, N, 2]` |
| `input_labels` | `int64` | `[B, 1, N]` |

Example:

```js
function makePointPrompt(points, resizedHeight, resizedWidth) {
  const coordinates = points.flatMap(({ x, y }) => [
    x * resizedWidth,
    y * resizedHeight,
  ]);
  const labels = points.map(({ label }) => BigInt(label));

  return {
    input_points: new Tensor(
      "float32",
      coordinates,
      [1, 1, points.length, 2],
    ),
    input_labels: new Tensor(
      "int64",
      labels,
      [1, 1, points.length],
    ),
  };
}
```

### Box tensor shape

| Name | Type | Shape |
|---|---|---|
| `input_boxes` | `float32` | `[B, M, 4]` |

For one box:

```js
function makeBoxPrompt(box, resizedHeight, resizedWidth) {
  const coordinates = [
    box.x1 * resizedWidth,
    box.y1 * resizedHeight,
    box.x2 * resizedWidth,
    box.y2 * resizedHeight,
  ];

  return {
    input_boxes: new Tensor("float32", coordinates, [1, 1, 4]),
  };
}
```

Transformers.js fills the required empty point or box tensors when only one
prompt type is provided. A direct ONNX Runtime caller must provide every graph
input, including correctly shaped empty tensors:

- point-only calls need an empty `input_boxes` tensor shaped `[B, 0, 4]`;
- box-only calls need empty point data shaped `[B, 1, 0, 2]` and matching
  empty labels;
- when points and boxes are supplied together, the Transformers.js wrapper
  requires exactly one box per image.

## Stage 5: run the prompt encoder and mask decoder

Merge the cached image embeddings with the current prompt:

```js
const pointPrompt = makePointPrompt(
  points,
  imageInputs.reshaped_input_sizes[0][0],
  imageInputs.reshaped_input_sizes[0][1],
);

const outputs = await model({
  ...imageEmbeddings,
  ...pointPrompt,
});
```

For a box prompt, spread `makeBoxPrompt(...)` instead.

The decoder graph consumes:

| Input | Type | Shape |
|---|---|---|
| `input_points` | `float32` | `[B, 1, N, 2]` |
| `input_labels` | `int64` | `[B, 1, N]` |
| `input_boxes` | `float32` | `[B, M, 4]` |
| `image_embeddings.0` | `float32` | `[B, 32, 288, 288]` |
| `image_embeddings.1` | `float32` | `[B, 64, 144, 144]` |
| `image_embeddings.2` | `float32` | `[B, 256, 72, 72]` |

It returns:

| Output | Type | Meaning |
|---|---|---|
| `iou_scores` | `float32` | Three predicted quality scores for each prompt result |
| `pred_masks` | `float32` | Three low-resolution mask-logit candidates per prompt result |
| `object_score_logits` | `float32` | Object-presence confidence logit |

The model configuration sets `num_multimask_outputs` to `3`. Treat
`pred_masks` as logits, not probabilities and not an already resized image.

## Stage 6: resize and threshold masks

Use the processor's post-processing method:

```js
const masks = await processor.post_process_masks(
  outputs.pred_masks,
  imageInputs.original_sizes,
  imageInputs.reshaped_input_sizes,
);
```

The inspected Transformers.js 3.8.0 post-processor performs this sequence:

1. bilinearly upsample low-resolution logits to the processor canvas;
2. crop to the reshaped image dimensions;
3. bilinearly resize to the original source image dimensions;
4. threshold logits at `0`;
5. return Boolean masks.

The default logit threshold is `0`, equivalent to a sigmoid probability
threshold of `0.5`.

The crop in step 2 is wrapper behavior based on processor metadata; it is not
evidence that the downloaded config requests a center crop. A direct ONNX
caller should reproduce the same resize/pad metadata it generated during
preprocessing and should not add an unconditional image crop.

Passing `original_sizes` and `reshaped_input_sizes` from the same processor
call is essential. Otherwise masks may be stretched or shifted.

## Stage 7: choose a mask candidate

For the common single-prompt case, select the candidate with the highest
predicted IoU:

```js
function argmax(values) {
  let best = 0;
  for (let index = 1; index < values.length; index += 1) {
    if (values[index] > values[best]) best = index;
  }
  return best;
}

const candidateIndex = argmax(outputs.iou_scores.data);
```

The playground uses this highest-IoU rule. In a multi-prompt or multi-box
application, preserve the batch and prompt-result axes and compute the best
candidate independently for each result rather than taking one global argmax.

`object_score_logits` can be converted with a sigmoid and used as an
additional rejection signal, but the playground's visible path chooses masks
using `iou_scores`.

## Stage 8: render or export

The selected Boolean mask has the original image dimensions. Common outputs
are:

- a translucent color overlay;
- an alpha-matted foreground PNG;
- a one-channel binary PNG;
- a binary array for later measurement.

For an overlay, set RGBA only where the selected mask is `true`. For a cutout,
copy source RGB and use the mask as alpha.

Keep manual paint edits separate from inference output so prompts can be
rerun without accumulating destructive changes.

## End-to-end pseudocode

```js
// One-time initialization.
const model = await Sam3TrackerModel.from_pretrained("sam3-q4", {
  device: "webgpu",
  dtype: {
    vision_encoder: "q4",
    prompt_encoder_mask_decoder: "fp32",
  },
});
const processor = await AutoProcessor.from_pretrained("sam3-q4");

// Once per image.
const image = await RawImage.fromURL(imageUrl);
const imageInputs = await processor(image);
const imageEmbeddings = await model.get_image_embeddings(imageInputs);

// Repeated for each prompt update.
const [resizedHeight, resizedWidth] =
  imageInputs.reshaped_input_sizes[0];
const prompt = makePointPrompt(points, resizedHeight, resizedWidth);

const {
  pred_masks,
  iou_scores,
  object_score_logits,
} = await model({
  ...imageEmbeddings,
  ...prompt,
});

const masks = await processor.post_process_masks(
  pred_masks,
  imageInputs.original_sizes,
  imageInputs.reshaped_input_sizes,
);

const candidateIndex = argmax(iou_scores.data);
const selectedMask = selectCandidate(masks[0], candidateIndex);
renderMask(selectedMask);
```

`selectCandidate` and `renderMask` are application-specific because tensor
indexing and output format depend on the chosen UI or image library.

## Performance model

Separate the pipeline into two timing domains:

```text
model load          once per application/runtime
image preprocessing once per image
vision encoding     once per image; expensive
prompt decoding     once per click/box update; comparatively cheap
mask rendering      once per prompt update
```

Cache only data associated with the current image:

- processor metadata;
- the three image embeddings;
- current prompt state;
- latest masks and scores.

Clear all four when the image changes. Do not rerun the vision encoder for
every click.

For responsive pointer interactions, allow at most one decoder invocation at
a time. If prompts change while decoding, keep only the newest pending prompt
state and run it after the current decode finishes.

## Validation checklist

Before trusting results, verify:

1. All six downloaded files are served and return HTTP success.
2. The two `.onnx_data` downloads are not HTML error pages or truncated.
3. WebGPU is available through `navigator.gpu`.
4. The encoder input is `float32 [1,3,1008,1008]`.
5. Embedding outputs match the three decoder shapes listed above.
6. Point and box coordinates are scaled by `reshaped_input_sizes`, not CSS
   display dimensions.
7. Positive labels are `1n`, negative labels are `0n`, and label tensors are
   `int64`.
8. Decoder masks are resized with the original and reshaped sizes from the
   same processor result.
9. Mask logits are thresholded after interpolation.
10. Candidate selection is performed independently per prompt result.
11. Changing prompts does not rerun image encoding.
12. Changing the image clears cached embeddings before the next decode.

Useful sanity tests:

- a positive point near the center of a clear object should select that
  object;
- adding a negative point on an unwanted region should remove or reduce it;
- a tight box should produce a mask contained mostly within that box;
- the final mask dimensions must exactly equal the original image dimensions;
- the same image and prompts should produce stable results across runs.

## Common failure modes

### Model loads but decoder fails

Usually one of the external-data files is missing, misplaced, truncated, or
served under a different filename. The ONNX graph expects its sidecar by the
literal relative filename stored in the graph.

### Mask is shifted or stretched

Coordinates were derived from the displayed element rather than normalized
image coordinates, width and height were swapped, or mask post-processing
used the wrong original/reshaped sizes.

### Positive and negative clicks behave identically

`input_labels` was created as a JavaScript number or `float32` tensor instead
of an `int64` tensor containing `BigInt` values.

### Segmentation works but is slow after every click

The vision encoder is being rerun. Cache `image_embeddings.0`,
`image_embeddings.1`, and `image_embeddings.2` until the source image changes.

### Browser reports no compatible backend

The reproduced playground path requires WebGPU. Confirm `navigator.gpu`,
current browser support, hardware acceleration, and a secure context
(`https://` or localhost).

### Text prompts do nothing

These exported graph inputs support points and boxes, not text. A text-driven
SAM3 pipeline requires additional model components or a different export.
