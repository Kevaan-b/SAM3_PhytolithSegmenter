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

On first startup, embeddings are written to `.samotator-cache/embeddings/`
while the selected image and its neighbors receive priority. For the current
238-image dataset this uses about 4.7 GiB. Later runs reuse the persistent
cache and warm it into an H100 LRU with a 16 GiB default budget. Override the
budget with `SAMOTATOR_GPU_CACHE_GIB`.

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
- **Rescan data folder** picks up added, modified, and removed files and queues
  new embeddings without restarting the service.

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
- Manual edits stay applied when SAM produces a newer mask and reset when you
  change images. Brush edits never call the H100 service.

The files under `sam3-q4/` must remain in their existing layout because both
ONNX graphs reference their adjacent `.onnx_data` files by name.
