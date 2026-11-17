# Samotator hover demo

Local point-prompt segmentation using the supplied SAM3 Q4 ONNX bundle and
an NVIDIA H100 inference service.

## Set up the H100 server

Run these steps on the Linux server with the NVIDIA H100. They require an
existing conda installation, internet access to conda-forge and the npm
registry, and an NVIDIA driver compatible with CUDA 12.9. The driver is a
system prerequisite; it is not installed by conda.

Create and activate a dedicated environment with Git and Git LFS. This avoids
installing tools into your conda `base` environment:

```bash
conda create -n samotator --override-channels -c conda-forge git git-lfs
conda activate samotator
```

For a new checkout, clone the repository and enter its directory:

```bash
git clone <repository-url>
cd samotator
```

For an existing checkout, enter the repository directory and pull changes:

```bash
cd /path/to/samotator
git pull --ff-only
```

Install the rest of the conda-forge dependencies and fetch the model sidecars:

```bash
conda env update -n samotator -f environment.yml
git lfs install --local
git lfs pull
```

The environment file selects a CUDA build of ONNX Runtime, CUDA 12.9, a
CPU build of PyTorch for mask postprocessing, Python, Node.js, and all Python
application dependencies. The H100 performs ONNX inference. Keep the two
`.onnx_data` files beside their matching `.onnx` graphs. The server can also
resolve Git LFS pointer files using objects already in the local LFS cache. It
does not download weights at startup; missing LFS objects cause a startup
error.

Install the JavaScript packages specified in the lockfile and build the frontend:
Node.js and npm come from conda, while the application-specific JavaScript
packages come from the npm registry.

```bash
npm ci
npm run build
```

Confirm that ONNX Runtime reports its CUDA provider:

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

The output should include `CUDAExecutionProvider`. This check confirms the
provider is present; the service verifies that the model can actually initialize
on CUDA when a user connects.

Start the shared service on the H100 server, from the repository directory:

```bash
conda activate samotator
python -m server.run
```

After pulling future changes, update the environment, model files, and frontend
before restarting the service:

```bash
conda activate samotator
conda env update -n samotator -f environment.yml
git lfs pull
npm ci
npm run build
```
 Run the Python tests with `python -m pytest -q server`
and the frontend tests with `npm test` while the environment is active.

The gateway serves the built frontend and listens on port 8000 by default. A
user on the trusted LAN or VPN can open `http://SERVER_ADDRESS:8000` in a
current Chrome or Edge browser and click **Connect**. To connect from a
workstation using VS Code Remote SSH, connect VS Code to the server, open the
**Ports** panel, forward remote port `8000`, then open the forwarded local
address (usually `http://localhost:8000`) in the workstation browser. Keep the
remote SSH connection and service running. The first connection starts a
single GPU worker and loads SAM3; by default up to eight browser sessions share
it. Disconnect saves pending changes and releases the session. The last disconnect stops the
worker and releases its GPU memory, while the lightweight gateway remains
available. Closing a browser or losing its connection releases its place and
image reservation after the lease expires. A browser refresh reuses its lease.
Run only one gateway process for this dataset; do not use multiple Uvicorn
workers or multiple gateway instances.

The gateway binds to `0.0.0.0:8000` by default. Set `SAMOTATOR_HOST` and
`SAMOTATOR_PORT` to change the address. `SAMOTATOR_MAX_SESSIONS` sets the
maximum number of connected browser tabs (default `8`); set it to a positive
integer before starting the gateway, for example
`SAMOTATOR_MAX_SESSIONS=12 python -m server.run`. The private GPU worker
listens on localhost port 18001; change that port with
`SAMOTATOR_WORKER_PORT`. Set `SAMOTATOR_LEASE_SECONDS` (default `60`) to adjust
failure recovery. Set `SAMOTATOR_GPU_CACHE_GIB` (default `16`) to adjust the
GPU embedding cache size. Restrict network access using a firewall, trusted
LAN, VPN, or SSH port forwarding; the application has no login or transport
encryption. The gateway must remain running, for example under a systemd
service; restarting it disconnects existing sessions. Reconnect in the browser
if a session expires. A session is a browser tab, and uses a slot until it
disconnects or its lease expires.

Only one person can edit an image at a time. Others see the saved version and
can use **Try editing** after it is released. Categories and annotation data are
shared. Embeddings persist under `.samotator-cache/embeddings/`, are reused
across sessions, and are computed folder-on-demand. The selected image and its
neighbors are prioritized ahead of other images. GPU embeddings share a 16 GiB
LRU cache, configurable with `SAMOTATOR_GPU_CACHE_GIB`; both the models and GPU
cache are freed when the worker exits. Interactive decoding takes priority over
background encoding. Requests currently run individually: batch execution is
not enabled until correctness and interactive latency have been benchmarked on
the deployed model and hardware.

The launcher discovers installed NVIDIA runtime libraries and requires ONNX
Runtime's CUDA provider; there is no CPU fallback. If the ONNX sidecars are Git
LFS pointer files, the server resolves their already-downloaded objects into
`.samotator-cache/model/`. Missing LFS objects produce a startup error. No
remote model requests are made.

For frontend development, run `npm run dev:server` and `npm run dev` in separate
terminals. Vite proxies `/api` to the gateway; the production gateway serves the
built frontend and images directly.

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
