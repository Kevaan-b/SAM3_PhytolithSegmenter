# Samotator hover demo

Local point-prompt segmentation using the supplied SAM3 Q4 ONNX bundle and
an NVIDIA H100 inference service.

## Run

```bash
npm install
python -m pip install -r requirements.txt
npm run dev:server
```

In a second terminal:

```bash
npm run dev
```

Open <http://127.0.0.1:5173> in a current Chrome or Edge browser.

The Python launcher discovers the installed NVIDIA runtime libraries and
starts FastAPI on `127.0.0.1:8000`. It requires ONNX Runtime's CUDA provider;
there is no slow CPU fallback. Vite proxies `/api` to that service.

Embeddings are computed folder-on-demand and persisted under `.samotator-cache/embeddings/`. The selected image is prepared first, then its immediate neighbors, then the rest of the selected folder. Previously visited folders resume at low priority; unvisited folders are never queued. Hovering or drawing pauses new background encodes. Existing embeddings are reused through a persistent fingerprint catalog without hashing every source image at startup.

Only the selected image and its neighbors are proactively kept on the H100. Other embeddings enter a 16 GiB GPU LRU when used, and duplicate image content shares the same GPU entry. Override the budget with `SAMOTATOR_GPU_CACHE_GIB`.

If the ONNX sidecars are Git LFS pointer files, the server resolves their
already-downloaded objects into `.samotator-cache/model/`. Missing LFS objects
produce a clear startup error. No remote model requests are made.

## Data folders

Put images anywhere under `data/`. The development server discovers nested
folders recursively without requiring a generated manifest:

```text
data/
├── train/
│   ├── image-001.png
│   └── image-002.png
├── val/
└── test/
```

The folder dropdown is an expandable tree that can jump to any depth, while
breadcrumbs move back up the hierarchy. The image selector contains only
images directly inside the current folder.

- Set the navigation toggle to **Folders** to make **Previous** and **Next**
  move between sibling folders at the current level.
- Set it to **Images** to move between images in the current folder.
- **Rescan data folder** picks up added, modified, and removed files. New
  embeddings are queued when their folder is selected.

Supported file extensions are PNG, JPEG, WebP, GIF, BMP, TIFF, and AVIF.

## Controls

- Move over the image for a temporary point preview.
- Choose **Positive** or **Negative** to change the temporary point type.
- Click to pin the current point.
- Use **Undo** or **Clear all** to revise pinned prompts.
- After a mask is generated, choose **Marker** or **Eraser** and hold-drag to
  refine it with independent source-pixel brush sizes.
- **Invert mask** shows the exact complement. **Undo edit** reverts one whole
  stroke or inversion, while **Reset edits** returns to the latest SAM mask.
- Manual edits stay applied when SAM produces a newer mask. They are flattened
  into the saved image draft across navigation; per-stroke undo history resets
  when the image is reloaded. Brush edits never call the H100 service.

## Classes and COCO annotations

Classes are defined in `data/metadata/categories.json`. The Masks popover can
add, rename, recolor, and remove classes, then create any number of independent
instances from the class list. Removed classes remain valid for annotations
that already reference their stable numeric ID.

Annotation changes autosave after a short delay and can also be saved with the
**Save** button or `Ctrl/Cmd+S`. Editable per-image drafts live under
`data/.samotator/annotations/`. Standards-compliant COCO instance segmentation files are regenerated atomically under `data/annotations/`, using compressed RLE masks with bounding boxes and areas derived from the final binary masks. Draft saves and statistics update immediately; COCO regeneration is coalesced in the background so autosave never rescans the complete dataset.

A rebuildable SQLite metadata index lives at `.samotator-cache/annotations.sqlite3`. The Statistics tab queries this index and requests at most eight cached 320 px WebP previews for the selected class. Rebuild the index from the editable drafts at any time with `python -m server.annotation_index rebuild`.

The files under `sam3-q4/` must remain in their existing layout because both
ONNX graphs reference their adjacent `.onnx_data` files by name.
