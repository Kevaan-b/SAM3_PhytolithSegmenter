import "./styles.css";

import {
  clearPoints,
  composePrompts,
  fitDimensions,
  normalizePointer,
  promptKey,
  removeLastPoint,
} from "./core";
import {
  adjacentFolder,
  adjacentImage,
  findFolder,
  firstFolderWithImages,
  flattenFolders,
  folderBreadcrumbs,
  type DataFolder,
  type DataImage,
  type NavigationMode,
} from "./data-navigator";
import type {
  BrushOperation,
  MainToWorkerMessage,
  MaskPoint,
  PointLabel,
  PointPrompt,
  WorkerToMainMessage,
} from "./protocol";
import { MaskLayerCollection, type MaskLayer } from "./mask-layers";

type ActiveTool = "positive" | "negative" | "marker" | "eraser";

document.querySelector<HTMLDivElement>("#app")!.innerHTML = `
  <div class="app-shell">
    <header class="masthead">
      <a class="brand" href="/" aria-label="Samotator home">
        <span class="brand-mark" aria-hidden="true">
          <span></span><span></span><span></span>
        </span>
        <span>Samotator</span>
      </a>
      <div class="model-pill">
        <span class="model-dot"></span>
        SAM3 Q4 · H100 CUDA
      </div>
    </header>

    <main class="workspace">
      <aside class="control-panel">
        <div class="eyebrow">Interactive segmentation</div>
        <h1>Point. Refine.<br /><em>Reveal.</em></h1>
        <p class="lede">
          Move across an image to preview a segment. Pin points to shape the
          mask—the image stays on your machine.
        </p>

        <section class="control-section">
          <div class="section-heading">
            <span>01</span>
            <h2>Browse data</h2>
          </div>
          <div class="data-browser">
            <nav class="folder-crumbs" id="folder-crumbs" aria-label="Data folder path"></nav>
            <span class="browser-label" id="folder-tree-label">Folder</span>
            <div class="folder-tree-picker" id="folder-tree-picker">
              <button
                class="folder-tree-trigger"
                id="folder-tree-trigger"
                type="button"
                aria-labelledby="folder-tree-label folder-tree-value"
                aria-haspopup="tree"
                aria-expanded="false"
                disabled
              >
                <span id="folder-tree-value">Scanning data/…</span>
                <span aria-hidden="true">⌄</span>
              </button>
              <div class="folder-tree-menu" id="folder-tree-menu" hidden>
                <div class="folder-tree" id="folder-tree" role="tree"></div>
              </div>
            </div>
            <label class="browser-label" for="image-select">Image</label>
            <select class="browser-select" id="image-select" disabled>
              <option>No images</option>
            </select>
            <div class="browser-navigation">
              <button class="nav-button" id="previous-button" type="button" disabled>
                ← Previous
              </button>
              <div class="navigation-toggle" role="group" aria-label="Previous and next target">
                <button class="active" id="navigate-folders" type="button" aria-pressed="true">
                  Folders
                </button>
                <button id="navigate-images" type="button" aria-pressed="false">
                  Images
                </button>
              </div>
              <button class="nav-button" id="next-button" type="button" disabled>
                Next →
              </button>
            </div>
            <button class="refresh-button" id="refresh-data" type="button">
              ↻ Rescan data folder
            </button>
            <p class="data-summary" id="data-summary">Looking for images under data/…</p>
          </div>
        </section>

        <section class="control-section masks-section">
          <div class="section-heading">
            <span>02</span>
            <h2>Masks</h2>
          </div>
          <div class="mask-picker" id="mask-picker">
            <div class="mask-picker-toolbar">
              <button class="mask-picker-trigger" id="mask-picker-trigger" type="button" aria-expanded="false" aria-haspopup="dialog">
                <span class="layer-swatch" id="active-layer-swatch"></span>
                <span class="active-layer-name" id="active-layer-name">Mask 1</span>
                <span class="mask-count" id="mask-count">1 mask</span>
                <span aria-hidden="true">⌄</span>
              </button>
              <button class="add-mask-button" id="add-mask" type="button" aria-label="Add mask layer">+</button>
            </div>
            <div class="mask-picker-menu" id="mask-picker-menu" hidden>
              <div class="new-instance-row">
                <select id="new-mask-class" aria-label="Class for new mask"></select>
                <button id="add-mask-confirm" type="button">Add instance</button>
              </div>
              <div class="mask-layer-list" id="mask-layer-list" role="listbox" aria-label="Mask layers"></div>
              <div class="mask-layer-editor">
                <label for="mask-layer-class">Class</label>
                <select id="mask-layer-class"></select>
                <label for="mask-layer-name">Name</label>
                <input id="mask-layer-name" maxlength="80" />
                <label for="mask-layer-color">Color</label>
                <input id="mask-layer-color" type="color" />
                <button class="new-class-button" id="new-class" type="button">New class</button>
                <button class="archive-class-button" id="archive-class" type="button">Remove class</button>
                <button class="delete-mask-button" id="delete-mask" type="button">Delete mask</button>
              </div>
              <label class="overlap-toggle" for="prevent-mask-overlap">
                <span>
                  <strong>Prevent overlaps</strong>
                  <small>Clip the latest mask against all others</small>
                </span>
                <input id="prevent-mask-overlap" type="checkbox" />
              </label>
            </div>
          </div>
        </section>

        <section class="control-section">
          <div class="section-heading">
            <span>03</span>
            <h2>Set point type</h2>
          </div>
          <div class="tool-switch" role="group" aria-label="Point type">
            <button class="tool-button active" id="positive-tool" type="button" aria-pressed="true">
              <span class="tool-icon positive">+</span>
              Positive
            </button>
            <button class="tool-button" id="negative-tool" type="button" aria-pressed="false">
              <span class="tool-icon negative">−</span>
              Negative
            </button>
          </div>
          <p class="tool-hint" id="tool-hint">
            Positive points include a region.
          </p>
        </section>

        <div class="point-actions">
          <button class="text-button" id="undo-button" type="button" disabled>
            <span aria-hidden="true">↶</span> Undo
          </button>
          <button class="text-button" id="clear-button" type="button" disabled>
            <span aria-hidden="true">×</span> Clear all
          </button>
          <span class="point-count" id="point-count">0 pinned</span>
        </div>

        <section class="control-section mask-edit-section">
          <div class="section-heading">
            <span>04</span>
            <h2>Refine mask</h2>
          </div>
          <div class="tool-switch" role="group" aria-label="Mask brush">
            <button class="tool-button" id="marker-tool" type="button" aria-pressed="false" disabled>
              <span class="tool-icon marker-icon">●</span>
              Marker
            </button>
            <button class="tool-button" id="eraser-tool" type="button" aria-pressed="false" disabled>
              <span class="tool-icon eraser-icon">○</span>
              Eraser
            </button>
          </div>
          <div class="brush-controls">
            <label for="marker-size">
              <span>Marker size</span><output id="marker-size-value">20 px</output>
            </label>
            <input id="marker-size" type="range" min="2" max="200" step="2" value="20" />
            <label for="eraser-size">
              <span>Eraser size</span><output id="eraser-size-value">32 px</output>
            </label>
            <input id="eraser-size" type="range" min="2" max="200" step="2" value="32" />
          </div>
          <div class="mask-actions">
            <button class="text-button" id="invert-mask" type="button" aria-pressed="false" disabled>◐ Invert mask</button>
            <button class="text-button" id="undo-edit" type="button" disabled>↶ Undo edit</button>
            <button class="text-button" id="reset-edits" type="button" disabled>× Reset edits</button>
          </div>
        </section>
      </aside>

      <section class="viewer-panel" aria-label="Interactive segmentation viewer">
        <div class="viewer-topline">
          <div>
            <div class="eyebrow">Live canvas</div>
            <h2 id="viewer-title">No image selected</h2>
          </div>
          <div class="status-chip loading" id="status-chip" role="status" aria-live="polite">
            <span class="status-light"></span>
            <span id="status-label">Connecting to H100…</span>
          </div>
          <div class="save-controls">
            <span id="save-status">Saved</span>
            <button id="save-annotations" type="button" disabled>Save</button>
          </div>
        </div>

        <div class="progress-track" id="progress-track" aria-hidden="true">
          <span id="progress-bar"></span>
        </div>

        <div class="image-frame" id="image-frame">
          <div class="image-stage disabled" id="image-stage">
            <img id="source-image" alt="Selected data image" draggable="false" />
            <canvas id="mask-overlay" aria-hidden="true"></canvas>
            <div class="marker-layer" id="marker-layer" aria-hidden="true"></div>
            <div class="brush-cursor" id="brush-cursor" aria-hidden="true"></div>
            <div class="stage-message" id="stage-message">
              <span class="loader"></span>
              <strong>Connecting to the H100</strong>
              <small>The server is loading SAM3 and preparing embeddings.</small>
            </div>
          </div>
          <div class="corner-guide top-left"></div>
          <div class="corner-guide top-right"></div>
          <div class="corner-guide bottom-left"></div>
          <div class="corner-guide bottom-right"></div>
        </div>

        <div class="viewer-footer">
          <p id="interaction-hint"><span class="cursor-symbol">⌖</span> Hover to preview · Click to pin</p>
          <dl class="metrics">
            <div><dt>Encoder</dt><dd id="encode-metric">—</dd></div>
            <div><dt>Last decode</dt><dd id="decode-metric">—</dd></div>
          </dl>
        </div>
      </section>
    </main>
  </div>
`;

const folderCrumbs = getElement<HTMLElement>("folder-crumbs");
const folderTreePicker = getElement<HTMLDivElement>("folder-tree-picker");
const folderTreeTrigger =
  getElement<HTMLButtonElement>("folder-tree-trigger");
const folderTreeValue = getElement<HTMLSpanElement>("folder-tree-value");
const folderTreeMenu = getElement<HTMLDivElement>("folder-tree-menu");
const folderTree = getElement<HTMLDivElement>("folder-tree");
const imageSelect = getElement<HTMLSelectElement>("image-select");
const previousButton = getElement<HTMLButtonElement>("previous-button");
const nextButton = getElement<HTMLButtonElement>("next-button");
const refreshDataButton = getElement<HTMLButtonElement>("refresh-data");
const navigateFolders = getElement<HTMLButtonElement>("navigate-folders");
const navigateImages = getElement<HTMLButtonElement>("navigate-images");
const dataSummary = getElement<HTMLParagraphElement>("data-summary");
const imageStage = getElement<HTMLDivElement>("image-stage");
const sourceImage = getElement<HTMLImageElement>("source-image");
const overlay = getElement<HTMLCanvasElement>("mask-overlay");
const markerLayer = getElement<HTMLDivElement>("marker-layer");
const maskPicker = getElement<HTMLDivElement>("mask-picker");
const maskPickerTrigger = getElement<HTMLButtonElement>("mask-picker-trigger");
const maskPickerMenu = getElement<HTMLDivElement>("mask-picker-menu");
const maskLayerList = getElement<HTMLDivElement>("mask-layer-list");
const activeLayerSwatch = getElement<HTMLSpanElement>("active-layer-swatch");
const activeLayerName = getElement<HTMLSpanElement>("active-layer-name");
const maskCount = getElement<HTMLSpanElement>("mask-count");
const addMaskButton = getElement<HTMLButtonElement>("add-mask");
const newMaskClass = getElement<HTMLSelectElement>("new-mask-class");
const addMaskConfirm = getElement<HTMLButtonElement>("add-mask-confirm");
const maskLayerClass = getElement<HTMLSelectElement>("mask-layer-class");
const maskLayerName = getElement<HTMLInputElement>("mask-layer-name");
const maskLayerColor = getElement<HTMLInputElement>("mask-layer-color");
const newClassButton = getElement<HTMLButtonElement>("new-class");
const archiveClassButton = getElement<HTMLButtonElement>("archive-class");
const deleteMaskButton = getElement<HTMLButtonElement>("delete-mask");
const preventMaskOverlap = getElement<HTMLInputElement>("prevent-mask-overlap");
const positiveTool = getElement<HTMLButtonElement>("positive-tool");
const negativeTool = getElement<HTMLButtonElement>("negative-tool");
const markerTool = getElement<HTMLButtonElement>("marker-tool");
const eraserTool = getElement<HTMLButtonElement>("eraser-tool");
const markerSize = getElement<HTMLInputElement>("marker-size");
const eraserSize = getElement<HTMLInputElement>("eraser-size");
const markerSizeValue = getElement<HTMLOutputElement>("marker-size-value");
const eraserSizeValue = getElement<HTMLOutputElement>("eraser-size-value");
const invertMaskButton = getElement<HTMLButtonElement>("invert-mask");
const undoEditButton = getElement<HTMLButtonElement>("undo-edit");
const resetEditsButton = getElement<HTMLButtonElement>("reset-edits");
const toolHint = getElement<HTMLParagraphElement>("tool-hint");
const undoButton = getElement<HTMLButtonElement>("undo-button");
const clearButton = getElement<HTMLButtonElement>("clear-button");
const pointCount = getElement<HTMLSpanElement>("point-count");
const viewerTitle = getElement<HTMLHeadingElement>("viewer-title");
const statusChip = getElement<HTMLDivElement>("status-chip");
const statusLabel = getElement<HTMLSpanElement>("status-label");
const progressTrack = getElement<HTMLDivElement>("progress-track");
const progressBar = getElement<HTMLSpanElement>("progress-bar");
const stageMessage = getElement<HTMLDivElement>("stage-message");
const imageFrame = getElement<HTMLDivElement>("image-frame");
const encodeMetric = getElement<HTMLElement>("encode-metric");
const decodeMetric = getElement<HTMLElement>("decode-metric");
const brushCursor = getElement<HTMLDivElement>("brush-cursor");
const interactionHint = getElement<HTMLParagraphElement>("interaction-hint");
const saveAnnotationsButton = getElement<HTMLButtonElement>("save-annotations");
const saveStatus = getElement<HTMLSpanElement>("save-status");

interface Category { id: number; name: string; supercategory: string; color: string; active: boolean }
interface SavedDraftLayer { layerId: string; annotationId: number; categoryId: number; rawMask: string; effectiveMask: string }
interface SavedDraft {
  imageId: string;
  width?: number;
  height?: number;
  layers: SavedDraftLayer[];
  latestMaskLayerId: string | null;
  preventOverlap: boolean;
}

let dataRoot: DataFolder | null = null;
let currentFolder: DataFolder | null = null;
let activeImage: DataImage | null = null;
let navigationMode: NavigationMode = "folder";
const expandedFolderPaths = new Set<string>([""]);
let activeTool: ActiveTool = "positive";
let markerDiameter = 20;
let eraserDiameter = 32;
const maskLayers = new MaskLayerCollection();
let categories: Category[] = [{ id: 1, name: "object", supercategory: "phytolith", color: "#4094dc", active: true }];
let preventOverlap = false;
let dirty = false;
let annotationRevision = 0;
let saving = false;
let lastSaveError: string | null = null;
let saveTimer = 0;
const snapshotResolvers = new Map<string, (message: Extract<WorkerToMainMessage, { type: "annotation-snapshot" }>) => void>();
let nextStrokeId = 0;
let activeStroke: {
  pointerId: number;
  strokeId: number;
  layerId: string;
  operation: BrushOperation;
  radius: number;
} | null = null;
let hoverPoint: PointPrompt | null = null;
let pointerInside = false;
let modelReady = false;
let imageReady = false;
let imageRevision = 0;
let animationFrame = 0;
let latestPointer: { clientX: number; clientY: number } | null = null;
let displayImageSize: { width: number; height: number } | null = null;

sourceImage.addEventListener("load", () => {
  if (sourceImage.naturalWidth > 0 && sourceImage.naturalHeight > 0) {
    setStageSize(sourceImage.naturalWidth, sourceImage.naturalHeight);
  }
});
window.addEventListener("resize", () => {
  if (displayImageSize) {
    setStageSize(displayImageSize.width, displayImageSize.height);
  }
});
renderDataBrowser();
renderMaskPicker();
updatePointControls();
updateEditingControls();

if (
  !("Worker" in window) ||
  !("transferControlToOffscreen" in overlay)
) {
  showFatal(
    "This demo needs Web Workers and OffscreenCanvas. Open it in a current Chrome or Edge browser.",
    "Browser unsupported",
  );
} else {
  const inferenceWorker = new Worker(
    new URL("./model.worker.ts", import.meta.url),
    { type: "module" },
  );

  inferenceWorker.onmessage = ({
    data,
  }: MessageEvent<WorkerToMainMessage>) => {
    handleWorkerMessage(data);
  };
  inferenceWorker.onerror = (event) => {
    showFatal(event.message || "The inference worker stopped unexpectedly.");
  };

  const offscreen = overlay.transferControlToOffscreen();
  postWorker({ type: "initialize", canvas: offscreen }, [offscreen]);
  void initializeProject();

  function postWorker(
    message: MainToWorkerMessage,
    transfer: Transferable[] = [],
  ): void {
    inferenceWorker.postMessage(message, transfer);
  }

  function handleWorkerMessage(message: WorkerToMainMessage): void {
    switch (message.type) {
      case "status": {
        setStatus("loading", message.message);
        if (message.progress !== undefined) {
          progressTrack.classList.add("visible");
          progressBar.style.width = `${message.progress}%`;
        } else {
          progressTrack.classList.remove("visible");
        }
        if (message.phase === "encoding-image") {
          showStageMessage(
            "Encoding this image once",
            "Hover decoding starts as soon as the embeddings are ready.",
          );
        }
        return;
      }

      case "model-ready": {
        modelReady = true;
        progressTrack.classList.remove("visible");
        if (activeImage) {
          loadActiveImage();
        } else {
          setStatus("ready", "Model ready · select an image");
        }
        return;
      }

      case "cache-status": {
        dataSummary.textContent = `${message.ready}/${message.total} embeddings cached · ${message.gpuResident} on H100 · ${message.queueDepth} queued`;
        if (message.total > 0 && message.ready < message.total) {
          progressTrack.classList.add("visible");
          progressBar.style.width = `${(message.ready / message.total) * 100}%`;
        } else if (modelReady) {
          progressTrack.classList.remove("visible");
        }
        return;
      }

      case "image-ready": {
        if (message.imageRevision !== imageRevision) return;
        setStageSize(message.width, message.height);
        imageReady = true;
        renderMaskPicker();
        updateSaveState();
        imageStage.classList.remove("disabled");
        stageMessage.classList.add("hidden");
        encodeMetric.textContent = message.cacheHit
          ? `cache hit · ${formatDuration(message.encodeMs)}`
          : formatDuration(message.encodeMs);
        setStatus("ready", "Ready · hover to segment");
        return;
      }

      case "mask-ready": {
        if (
          message.imageRevision !== imageRevision ||
          !message.applied
        ) {
          return;
        }
        decodeMetric.textContent = `${formatDuration(message.decodeMs)} · GPU ${formatDuration(message.serverDecodeMs)}`;
        setStatus("ready", "Ready · hover to segment");
        return;
      }

      case "overlay-cleared":
        return;

      case "edit-state": {
        if (message.imageRevision !== imageRevision) return;
        const layer = maskLayers.get(message.layerId);
        if (!layer) return;
        layer.editState = {
          hasMask: message.hasMask,
          hasEdits: message.hasEdits,
          canUndo: message.canUndo,
          inverted: message.inverted,
        };
        if (layer.id !== maskLayers.active().id) return;
        updateEditingControls();
        return;
      }

      case "annotation-snapshot": {
        if (message.imageRevision !== imageRevision) return;
        snapshotResolvers.get(message.requestId)?.(message);
        snapshotResolvers.delete(message.requestId);
        return;
      }

      case "error": {
        if (
          message.imageRevision !== undefined &&
          message.imageRevision !== imageRevision
        ) {
          return;
        }
        showFatal(message.message);
      }
    }
  }

  async function loadActiveImage(): Promise<void> {
    if (!modelReady || !activeImage) return;
    const revision = imageRevision;
    const image = activeImage;
    imageReady = false;
    renderMaskPicker();
    imageStage.classList.add("disabled");
    setStatus("loading", "Encoding image…");
    showStageMessage(
      "Encoding this image once",
      "The cached embeddings make every point prompt much faster.",
    );
    let draft: SavedDraft;
    try {
      const response = await fetch(`/api/images/${encodeURIComponent(image.id)}/annotations`, { cache: "no-store" });
      if (!response.ok) throw await responseErrorFromFetch(response);
      draft = await response.json() as SavedDraft;
    } catch (error) {
      showFatal(error instanceof Error ? error.message : String(error), "Unable to load annotations");
      return;
    }
    if (revision !== imageRevision || activeImage?.id !== image.id) return;
    applyDraft(draft);
    postWorker({
      type: "load-image",
      imageRevision: revision,
      imageId: image.id,
      url: image.url,
      layers: maskLayers.all().map(({ id, color, visible }) => ({ id, color, visible })),
      activeLayerId: maskLayers.active().id,
      preventOverlap,
      restoredMasks: draft.layers.map((layer) => ({ layerId: layer.layerId, mask: base64ToBytes(layer.rawMask) })),
      latestMaskLayerId: draft.latestMaskLayerId ?? undefined,
    });
  }

  async function initializeProject(): Promise<void> {
    try {
      await refreshClasses();
      await refreshData();
    } catch (error) {
      showFatal(error instanceof Error ? error.message : String(error), "Unable to initialize annotations");
    }
  }

  async function refreshClasses(): Promise<void> {
    const response = await fetch("/api/classes", { cache: "no-store" });
    if (!response.ok) throw await responseErrorFromFetch(response);
    const document = await response.json() as { categories: Category[] };
    categories = document.categories;
    renderMaskPicker();
  }

  function applyDraft(draft: SavedDraft): void {
    preventOverlap = draft.preventOverlap;
    if (draft.layers.length > 0) {
      maskLayers.replace(draft.layers.map((layer) => {
        const category = categoryById(layer.categoryId);
        return {
          id: layer.layerId,
          annotationId: layer.annotationId,
          categoryId: layer.categoryId,
          name: category?.name ?? `Unknown class ${layer.categoryId}`,
          color: category?.color ?? "#8a8a8a",
        };
      }));
    } else {
      const category = activeCategories()[0] ?? categories[0] ?? defaultCategory();
      maskLayers.reset(category.id, category.name, category.color);
    }
    dirty = false;
    annotationRevision = 0;
    updateSaveState();
    renderMaskPicker();
  }

  async function saveAnnotations(): Promise<boolean> {
    if (!activeImage || !imageReady || saving) return !dirty;
    saving = true;
    lastSaveError = null;
    updateSaveState();
    const requestId = crypto.randomUUID?.() ?? `save-${Date.now()}`;
    const revision = imageRevision;
    const savedAnnotationRevision = annotationRevision;
    let snapshotTimeout = 0;
    const snapshot = new Promise<Extract<WorkerToMainMessage, { type: "annotation-snapshot" }>>((resolve, reject) => {
      snapshotResolvers.set(requestId, (message) => {
        window.clearTimeout(snapshotTimeout);
        resolve(message);
      });
      snapshotTimeout = window.setTimeout(() => {
        snapshotResolvers.delete(requestId);
        reject(new Error("The mask worker did not finish the save snapshot."));
      }, 15000);
    });
    postWorker({ type: "snapshot-annotations", requestId, imageRevision: revision });
    try {
      const result = await snapshot;
      if (revision !== imageRevision || !activeImage) return false;
      const response = await fetch(`/api/images/${encodeURIComponent(activeImage.id)}/annotations`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          width: result.width,
          height: result.height,
          latestMaskLayerId: result.latestMaskLayerId || null,
          preventOverlap,
          layers: result.layers.map((mask) => ({
            layerId: mask.layerId,
            categoryId: maskLayers.get(mask.layerId)?.categoryId,
            rawMask: bytesToBase64(mask.rawMask),
            effectiveMask: bytesToBase64(mask.effectiveMask),
          })),
        }),
      });
      if (!response.ok) throw await responseErrorFromFetch(response);
      dirty = annotationRevision !== savedAnnotationRevision;
      lastSaveError = null;
      return true;
    } catch (error) {
      lastSaveError = error instanceof Error ? error.message : "Unknown save error";
      return false;
    } finally {
      saving = false;
      updateSaveState();
      if (dirty && !lastSaveError) {
        window.clearTimeout(saveTimer);
        saveTimer = window.setTimeout(() => void saveAnnotations(), 400);
      }
    }
  }

  function markDirty(): void {
    dirty = true;
    annotationRevision += 1;
    updateSaveState();
    window.clearTimeout(saveTimer);
    saveTimer = window.setTimeout(() => void saveAnnotations(), 1200);
  }

  function updateSaveState(): void {
    saveAnnotationsButton.disabled = !activeImage || !imageReady || saving || !dirty;
    if (saving) saveStatus.textContent = "Saving…";
    else if (lastSaveError) saveStatus.textContent = `Save failed: ${lastSaveError}`;
    else if (dirty) saveStatus.textContent = "Unsaved";
    else saveStatus.textContent = "Saved";
  }

  saveAnnotationsButton.addEventListener("click", () => void saveAnnotations());
  window.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
      event.preventDefault();
      void saveAnnotations();
    }
  });
  window.addEventListener("beforeunload", (event) => {
    if (!dirty) return;
    event.preventDefault();
  });

  function submitPrompts(points: PointPrompt[], preview = false): void {
    const layer = maskLayers.active();
    if (!imageReady || !layer.visible) return;
    const key = promptKey(points);
    if (key === layer.lastPromptKey) return;
    layer.lastPromptKey = key;
    layer.stateRevision += 1;

    if (points.length === 0) {
      postWorker({
        type: "clear",
        imageRevision,
        layerId: layer.id,
        stateRevision: layer.stateRevision,
      });
      return;
    }

    postWorker({
      type: "decode",
      imageRevision,
      layerId: layer.id,
      stateRevision: layer.stateRevision,
      points,
      preview,
    });
  }

  imageStage.addEventListener("pointerenter", (event) => {
    if (!imageReady || !maskLayers.active().visible) return;
    pointerInside = true;
    if (isBrushTool(activeTool)) {
      updateBrushCursor(event.clientX, event.clientY);
    } else {
      queuePointer(event.clientX, event.clientY);
    }
  });

  imageStage.addEventListener("pointermove", (event) => {
    if (!imageReady || !maskLayers.active().visible) return;
    pointerInside = true;
    if (isBrushTool(activeTool)) {
      updateBrushCursor(event.clientX, event.clientY);
      if (activeStroke?.pointerId === event.pointerId) {
        postBrush("continue", activeStroke.strokeId, brushPoints(event));
      }
    } else {
      queuePointer(event.clientX, event.clientY);
    }
  });

  imageStage.addEventListener("pointerleave", () => {
    pointerInside = false;
    latestPointer = null;
    hoverPoint = null;
    removeHoverMarker();
    hideBrushCursor();
    if (isBrushTool(activeTool)) return;
    const layer = maskLayers.active();
    layer.lastPromptKey = null;
    submitPrompts(layer.pinnedPoints);
  });

  imageStage.addEventListener("pointerdown", (event) => {
    if (
      !imageReady ||
      !maskLayers.active().editState.hasMask ||
      !maskLayers.active().visible ||
      !isBrushTool(activeTool) ||
      event.button !== 0
    ) return;
    event.preventDefault();
    imageStage.setPointerCapture?.(event.pointerId);
    const layer = maskLayers.active();
    layer.editRevision += 1;
    const strokeId = ++nextStrokeId;
    activeStroke = {
      pointerId: event.pointerId,
      strokeId,
      layerId: layer.id,
      operation: brushOperation(activeTool),
      radius: activeBrushDiameter() / 2,
    };
    postBrush("begin", strokeId, [normalizeEventPoint(event)]);
  });

  imageStage.addEventListener("pointerup", finishBrushStroke);
  imageStage.addEventListener("pointercancel", finishBrushStroke);

  imageStage.addEventListener("click", (event) => {
    if (!imageReady || !maskLayers.active().visible || !pointerInside || !isPointTool(activeTool)) return;
    const normalized = normalizePointer(
      event.clientX,
      event.clientY,
      imageStage.getBoundingClientRect(),
    );
    const layer = maskLayers.active();
    layer.pinnedPoints.push({ ...normalized, label: pointLabel(activeTool) });
    hoverPoint = null;
    latestPointer = null;
    layer.lastPromptKey = null;
    renderMarkers();
    updatePointControls();
    submitPrompts(layer.pinnedPoints);
    markDirty();
  });

  function queuePointer(clientX: number, clientY: number): void {
    latestPointer = { clientX, clientY };
    if (animationFrame !== 0) return;

    animationFrame = requestAnimationFrame(() => {
      animationFrame = 0;
      if (
        !latestPointer ||
        !pointerInside ||
        !imageReady ||
        !isPointTool(activeTool) ||
        !maskLayers.active().visible
      ) return;
      const normalized = normalizePointer(
        latestPointer.clientX,
        latestPointer.clientY,
        imageStage.getBoundingClientRect(),
      );
      hoverPoint = { ...normalized, label: pointLabel(activeTool) };
      updateHoverMarker(hoverPoint);
      submitPrompts([...maskLayers.active().pinnedPoints, hoverPoint], true);
    });
  }

  function finishBrushStroke(event: PointerEvent): void {
    if (!activeStroke || activeStroke.pointerId !== event.pointerId) return;
    postBrush("end", activeStroke.strokeId, [normalizeEventPoint(event)]);
    imageStage.releasePointerCapture?.(event.pointerId);
    activeStroke = null;
    markDirty();
  }

  function postBrush(
    phase: "begin" | "continue" | "end",
    strokeId: number,
    points: MaskPoint[],
  ): void {
    if (!activeStroke || activeStroke.strokeId !== strokeId) return;
    postWorker({
      type: "brush",
      imageRevision,
      layerId: activeStroke.layerId,
      editRevision: maskLayers.get(activeStroke.layerId)?.editRevision ?? 0,
      strokeId,
      phase,
      operation: activeStroke.operation,
      radius: activeStroke.radius,
      points,
    });
  }

  function brushPoints(event: PointerEvent): MaskPoint[] {
    const events = event.getCoalescedEvents?.() ?? [event];
    return events.map(normalizeEventPoint);
  }

  function normalizeEventPoint(event: PointerEvent): MaskPoint {
    return normalizePointer(
      event.clientX,
      event.clientY,
      imageStage.getBoundingClientRect(),
    );
  }

  async function refreshData(): Promise<void> {
    refreshDataButton.disabled = true;
    dataSummary.textContent = "Scanning data/…";

    try {
      const response = await fetch("/api/data-tree?refresh=true", {
        cache: "no-store",
      });
      if (!response.ok) {
        throw new Error(`Data scan failed with HTTP ${response.status}.`);
      }

      const nextRoot = (await response.json()) as DataFolder;
      const previousFolderPath = currentFolder?.path;
      const previousImagePath = activeImage?.path;
      dataRoot = nextRoot;

      const nextFolder =
        (previousFolderPath
          ? findFolder(nextRoot, previousFolderPath)
          : undefined) ??
        firstFolderWithImages(nextRoot) ??
        nextRoot;
      const preferredImage = nextFolder.images.find(
        ({ path }) => path === previousImagePath,
      );
      selectFolder(
        nextFolder,
        preferredImage?.path,
      );

      const folderCount = flattenFolders(nextRoot).length - 1;
      const imageCount = countImages(nextRoot);
      dataSummary.textContent = `${imageCount} image${
        imageCount === 1 ? "" : "s"
      } in ${folderCount} folder${folderCount === 1 ? "" : "s"}.`;
    } catch (error) {
      dataRoot = null;
      currentFolder = null;
      activeImage = null;
      renderDataBrowser();
      dataSummary.textContent =
        error instanceof Error ? error.message : String(error);
      setStatus("error", "Unable to scan data/");
      showNotice(
        "Unable to read data/",
        "Check the terminal running Vite, then rescan the folder.",
      );
    } finally {
      refreshDataButton.disabled = false;
    }
  }

  function selectFolder(
    folder: DataFolder,
    preferredImagePath?: string,
  ): void {
    void selectFolderAfterSave(folder, preferredImagePath);
  }

  async function selectFolderAfterSave(folder: DataFolder, preferredImagePath?: string): Promise<void> {
    if (dirty && !(await saveAnnotations())) return;
    currentFolder = folder;
    expandAncestors(folder.path);
    const image =
      folder.images.find(({ path }) => path === preferredImagePath) ??
      folder.images[0] ??
      null;

    if (image) {
      selectImage(image);
    } else {
      clearActiveImage();
      renderDataBrowser();
    }
  }

  function selectImage(image: DataImage): void {
    void selectImageAfterSave(image);
  }

  async function selectImageAfterSave(image: DataImage): Promise<void> {
    if (activeImage?.path === image.path) {
      renderDataBrowser();
      return;
    }
    if (dirty && !(await saveAnnotations())) {
      renderDataBrowser();
      return;
    }

    activeImage = image;
    imageRevision += 1;
    imageReady = false;
    resetMaskUiState();
    hoverPoint = null;
    pointerInside = false;
    latestPointer = null;
    encodeMetric.textContent = "—";
    decodeMetric.textContent = "—";
    setDataImage(image);
    renderDataBrowser();
    renderMarkers();
    updatePointControls();
    loadActiveImage();
  }

  function clearActiveImage(): void {
    activeImage = null;
    imageRevision += 1;
    resetMaskUiState();
    imageReady = false;
    hoverPoint = null;
    pointerInside = false;
    latestPointer = null;
    displayImageSize = null;
    sourceImage.removeAttribute("src");
    sourceImage.alt = "No image selected";
    viewerTitle.textContent = currentFolder?.name ?? "No image selected";
    encodeMetric.textContent = "—";
    decodeMetric.textContent = "—";
    imageFrame.style.width = "100%";
    imageStage.style.width = "100%";
    imageStage.style.height = "420px";
    imageStage.style.aspectRatio = "auto";
    imageStage.classList.add("disabled");
    dirty = false;
    annotationRevision = 0;
    updateSaveState();
    renderMarkers();
    updatePointControls();
    postWorker({
      type: "clear",
      imageRevision,
      layerId: maskLayers.active().id,
      stateRevision: maskLayers.active().stateRevision,
    });
    showNotice(
      "No images in this folder",
      "Choose another folder or add supported image files directly inside it.",
    );
    setStatus("ready", "Folder selected · no images");
  }

  function moveSelection(offset: -1 | 1): void {
    if (!dataRoot || !currentFolder) return;
    if (navigationMode === "folder") {
      const folder = adjacentFolder(dataRoot, currentFolder, offset);
      if (folder) selectFolder(folder);
      return;
    }

    const image = adjacentImage(currentFolder, activeImage, offset);
    if (image) selectImage(image);
  }

  folderTreeTrigger.addEventListener("click", () => {
    setFolderTreeOpen(folderTreeMenu.hidden);
  });

  folderTree.addEventListener("click", (event) => {
    if (!dataRoot) return;
    const target = event.target as HTMLElement;
    const expandButton =
      target.closest<HTMLButtonElement>("[data-expand-folder]");
    if (expandButton) {
      const path = expandButton.dataset.expandFolder ?? "";
      if (expandedFolderPaths.has(path)) {
        expandedFolderPaths.delete(path);
      } else {
        expandedFolderPaths.add(path);
      }
      renderFolderTree();
      return;
    }

    const folderButton =
      target.closest<HTMLButtonElement>("[data-folder-path]");
    if (!folderButton) return;
    const folder = findFolder(
      dataRoot,
      folderButton.dataset.folderPath ?? "",
    );
    if (folder) {
      selectFolder(folder);
      setFolderTreeOpen(false);
    }
  });

  imageSelect.addEventListener("change", () => {
    const image = currentFolder?.images.find(
      ({ path }) => path === imageSelect.value,
    );
    if (image) selectImage(image);
  });

  navigateFolders.addEventListener("click", () => {
    setNavigationMode("folder");
  });
  navigateImages.addEventListener("click", () => {
    setNavigationMode("image");
  });
  previousButton.addEventListener("click", () => moveSelection(-1));
  nextButton.addEventListener("click", () => moveSelection(1));
  refreshDataButton.addEventListener("click", () => void refreshData());

  folderCrumbs.addEventListener("click", (event) => {
    if (!dataRoot) return;
    const button = (event.target as HTMLElement).closest<HTMLButtonElement>(
      "[data-folder-path]",
    );
    if (!button) return;
    const folder = findFolder(
      dataRoot,
      button.dataset.folderPath ?? "",
    );
    if (folder) selectFolder(folder);
  });

  document.addEventListener("pointerdown", (event) => {
    if (
      !folderTreeMenu.hidden &&
      !folderTreePicker.contains(event.target as Node)
    ) {
      setFolderTreeOpen(false);
    }
    if (!maskPickerMenu.hidden && !maskPicker.contains(event.target as Node)) {
      setMaskPickerOpen(false);
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      setFolderTreeOpen(false);
      setMaskPickerOpen(false);
    }
  });

  maskPickerTrigger.addEventListener("click", () => setMaskPickerOpen(maskPickerMenu.hidden));
  addMaskButton.addEventListener("click", () => {
    if (!imageReady) return;
    setMaskPickerOpen(true);
    newMaskClass.focus();
  });
  addMaskConfirm.addEventListener("click", () => {
    if (!imageReady) return;
    const category = categoryById(Number(newMaskClass.value));
    if (!category) return;
    cancelTransientInteraction();
    const layer = maskLayers.add(category.id, category.name, category.color);
    postWorker({ type: "create-layer", imageRevision, layer: descriptor(layer) });
    postWorker({ type: "activate-layer", imageRevision, layerId: layer.id });
    refreshActiveLayerUi();
    markDirty();
  });
  maskLayerList.addEventListener("click", (event) => {
    const target = event.target as HTMLElement;
    const visibility = target.closest<HTMLButtonElement>("[data-layer-visible]");
    if (visibility) {
      const id = visibility.dataset.layerVisible!;
      const layer = maskLayers.get(id);
      if (!layer) return;
      const visible = !layer.visible;
      maskLayers.setVisible(id, visible);
      if (id === maskLayers.active().id && !visible) cancelTransientInteraction();
      postWorker({ type: "update-layer", imageRevision, layerId: id, visible });
      refreshActiveLayerUi();
      return;
    }
    const row = target.closest<HTMLButtonElement>("[data-layer-select]");
    if (!row) return;
    cancelTransientInteraction();
    const layer = maskLayers.select(row.dataset.layerSelect!);
    postWorker({ type: "activate-layer", imageRevision, layerId: layer.id });
    refreshActiveLayerUi();
  });
  maskLayerClass.addEventListener("change", () => {
    const category = categoryById(Number(maskLayerClass.value));
    if (!category) return;
    const layer = maskLayers.active();
    maskLayers.setCategory(layer.id, category.id, category.name, category.color);
    postWorker({ type: "update-layer", imageRevision, layerId: layer.id, color: category.color });
    renderMaskPicker();
    markDirty();
  });
  maskLayerName.addEventListener("change", () => {
    void updateCategory(maskLayers.active().categoryId, { name: maskLayerName.value });
  });
  maskLayerColor.addEventListener("change", () => {
    void updateCategory(maskLayers.active().categoryId, { color: maskLayerColor.value });
  });
  newClassButton.addEventListener("click", () => {
    const name = window.prompt("New class name");
    if (name?.trim()) void addCategory(name.trim());
  });
  archiveClassButton.addEventListener("click", () => {
    const category = categoryById(maskLayers.active().categoryId);
    if (category && window.confirm(`Remove “${category.name}” from new mask choices? Existing masks will keep it.`)) {
      void archiveCategory(category.id);
    }
  });
  deleteMaskButton.addEventListener("click", () => {
    if (maskLayers.all().length === 1) return;
    cancelTransientInteraction();
    const deletedId = maskLayers.active().id;
    const active = maskLayers.delete(deletedId);
    postWorker({ type: "delete-layer", imageRevision, layerId: deletedId });
    postWorker({ type: "activate-layer", imageRevision, layerId: active.id });
    refreshActiveLayerUi();
    markDirty();
  });
  preventMaskOverlap.addEventListener("change", () => {
    preventOverlap = preventMaskOverlap.checked;
    postWorker({
      type: "set-overlap-prevention",
      imageRevision,
      enabled: preventOverlap,
      activeLayerId: maskLayers.active().id,
    });
    markDirty();
  });

  async function addCategory(name: string): Promise<void> {
    const response = await fetch("/api/classes", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, supercategory: "phytolith" }),
    });
    if (!response.ok) { window.alert((await responseErrorFromFetch(response)).message); return; }
    const category = await response.json() as Category;
    categories.push(category);
    renderMaskPicker();
    newMaskClass.value = String(category.id);
  }

  async function updateCategory(id: number, changes: Partial<Category>): Promise<void> {
    const response = await fetch(`/api/classes/${id}`, {
      method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(changes),
    });
    if (!response.ok) { window.alert((await responseErrorFromFetch(response)).message); renderMaskPicker(); return; }
    const updated = await response.json() as Category;
    categories = categories.map((item) => item.id === id ? updated : item);
    for (const layer of maskLayers.all()) {
      if (layer.categoryId !== id) continue;
      maskLayers.setCategory(layer.id, id, updated.name, updated.color);
      postWorker({ type: "update-layer", imageRevision, layerId: layer.id, color: updated.color });
    }
    renderMaskPicker();
  }

  async function archiveCategory(id: number): Promise<void> {
    const response = await fetch(`/api/classes/${id}`, { method: "DELETE" });
    if (!response.ok) { window.alert((await responseErrorFromFetch(response)).message); return; }
    const archived = await response.json() as Category;
    categories = categories.map((item) => item.id === id ? archived : item);
    renderMaskPicker();
  }

  function cancelTransientInteraction(): void {
    const layer = maskLayers.active();
    if (hoverPoint || latestPointer) {
      layer.stateRevision += 1;
      layer.lastPromptKey = null;
      postWorker({
        type: "cancel-preview",
        imageRevision,
        layerId: layer.id,
        stateRevision: layer.stateRevision,
      });
    }
    hoverPoint = null;
    latestPointer = null;
    pointerInside = false;
    activeStroke = null;
    removeHoverMarker();
    hideBrushCursor();
  }

  function refreshActiveLayerUi(): void {
    renderMaskPicker();
    renderMarkers();
    updatePointControls();
    updateEditingControls();
  }

  function setNavigationMode(mode: NavigationMode): void {
    navigationMode = mode;
    renderDataBrowser();
  }

  positiveTool.addEventListener("click", () => {
    setTool("positive");
    if (hoverPoint) {
      maskLayers.active().lastPromptKey = null;
      submitPrompts(composePrompts(maskLayers.active().pinnedPoints, hoverPoint), true);
    }
  });
  negativeTool.addEventListener("click", () => {
    setTool("negative");
    if (hoverPoint) {
      maskLayers.active().lastPromptKey = null;
      submitPrompts(composePrompts(maskLayers.active().pinnedPoints, hoverPoint), true);
    }
  });
  markerTool.addEventListener("click", () => setTool("marker"));
  eraserTool.addEventListener("click", () => setTool("eraser"));

  markerSize.addEventListener("input", () => {
    markerDiameter = Number(markerSize.value);
    markerSizeValue.value = `${markerDiameter} px`;
    resizeBrushCursor();
  });
  eraserSize.addEventListener("input", () => {
    eraserDiameter = Number(eraserSize.value);
    eraserSizeValue.value = `${eraserDiameter} px`;
    resizeBrushCursor();
  });

  invertMaskButton.addEventListener("click", () => {
    const layer = maskLayers.active();
    if (!layer.editState.hasMask || !layer.visible) return;
    layer.editRevision += 1;
    postWorker({ type: "invert-mask", imageRevision, layerId: layer.id, editRevision: layer.editRevision });
    markDirty();
  });
  undoEditButton.addEventListener("click", () => {
    const layer = maskLayers.active();
    if (!layer.editState.canUndo || !layer.visible) return;
    layer.editRevision += 1;
    postWorker({ type: "undo-edit", imageRevision, layerId: layer.id, editRevision: layer.editRevision });
    markDirty();
  });
  resetEditsButton.addEventListener("click", () => {
    const layer = maskLayers.active();
    if (!layer.editState.hasEdits || !layer.visible) return;
    layer.editRevision += 1;
    postWorker({ type: "reset-edits", imageRevision, layerId: layer.id, editRevision: layer.editRevision });
    markDirty();
  });

  undoButton.addEventListener("click", () => {
    const layer = maskLayers.active();
    layer.pinnedPoints = removeLastPoint(layer.pinnedPoints);
    hoverPoint = null;
    latestPointer = null;
    removeHoverMarker();
    layer.lastPromptKey = null;
    renderMarkers();
    updatePointControls();
    submitPrompts(layer.pinnedPoints);
    markDirty();
  });

  clearButton.addEventListener("click", () => {
    const layer = maskLayers.active();
    layer.pinnedPoints = clearPoints();
    hoverPoint = null;
    latestPointer = null;
    layer.lastPromptKey = null;
    renderMarkers();
    updatePointControls();
    submitPrompts([]);
    markDirty();
  });
}

function setTool(tool: ActiveTool): void {
  const layer = maskLayers.active();
  if (isBrushTool(tool) && (!layer.editState.hasMask || !layer.visible)) return;
  activeTool = tool;
  const controls: Array<[HTMLButtonElement, ActiveTool]> = [
    [positiveTool, "positive"],
    [negativeTool, "negative"],
    [markerTool, "marker"],
    [eraserTool, "eraser"],
  ];
  controls.forEach(([button, value]) => {
    button.classList.toggle("active", tool === value);
    button.setAttribute("aria-pressed", String(tool === value));
  });
  toolHint.textContent =
    tool === "positive"
      ? "Positive points include a region."
      : tool === "negative"
        ? "Negative points exclude a region."
        : tool === "marker"
          ? "Drag to add visible mask pixels."
          : "Drag to erase visible mask pixels.";
  interactionHint.innerHTML = isBrushTool(tool)
    ? '<span class="cursor-symbol">◯</span> Hold and drag to refine the mask'
    : '<span class="cursor-symbol">⌖</span> Hover to preview · Click to pin';
  imageStage.classList.toggle("brush-mode", isBrushTool(tool));
  brushCursor.classList.toggle("eraser", tool === "eraser");

  if (isPointTool(tool) && hoverPoint) {
    hoverPoint = { ...hoverPoint, label: pointLabel(tool) };
    layer.lastPromptKey = null;
    updateHoverMarker(hoverPoint);
  } else if (isBrushTool(tool)) {
    hoverPoint = null;
    latestPointer = null;
    removeHoverMarker();
  } else {
    hideBrushCursor();
  }
  updateEditingControls();
}

function isPointTool(tool: ActiveTool): tool is "positive" | "negative" {
  return tool === "positive" || tool === "negative";
}

function isBrushTool(tool: ActiveTool): tool is "marker" | "eraser" {
  return tool === "marker" || tool === "eraser";
}

function pointLabel(tool: "positive" | "negative"): PointLabel {
  return tool === "positive" ? 1 : 0;
}

function brushOperation(tool: "marker" | "eraser"): BrushOperation {
  return tool === "marker" ? "add" : "erase";
}

function activeBrushDiameter(): number {
  return activeTool === "eraser" ? eraserDiameter : markerDiameter;
}

function updateBrushCursor(clientX: number, clientY: number): void {
  const layer = maskLayers.active();
  if (!isBrushTool(activeTool) || !layer.editState.hasMask || !layer.visible || !displayImageSize) {
    hideBrushCursor();
    return;
  }
  const rect = imageStage.getBoundingClientRect();
  const point = normalizePointer(clientX, clientY, rect);
  brushCursor.style.left = `${point.x * 100}%`;
  brushCursor.style.top = `${point.y * 100}%`;
  resizeBrushCursor();
  brushCursor.classList.add("visible");
}

function resizeBrushCursor(): void {
  if (!displayImageSize || !isBrushTool(activeTool)) return;
  const rect = imageStage.getBoundingClientRect();
  const displayDiameter =
    activeBrushDiameter() * (rect.width / displayImageSize.width);
  brushCursor.style.width = `${displayDiameter}px`;
  brushCursor.style.height = `${displayDiameter}px`;
}

function hideBrushCursor(): void {
  brushCursor.classList.remove("visible");
}

function updateEditingControls(): void {
  const layer = maskLayers.active();
  const editable = layer.visible && layer.editState.hasMask;
  if (!editable && isBrushTool(activeTool)) setTool("positive");
  markerTool.disabled = !editable;
  eraserTool.disabled = !editable;
  markerSize.disabled = !editable;
  eraserSize.disabled = !editable;
  invertMaskButton.disabled = !editable;
  undoEditButton.disabled = !layer.visible || !layer.editState.canUndo;
  resetEditsButton.disabled = !layer.visible || !layer.editState.hasEdits;
  invertMaskButton.classList.toggle("active", layer.editState.inverted);
  invertMaskButton.setAttribute("aria-pressed", String(layer.editState.inverted));
  clearButton.disabled = !layer.visible || (layer.pinnedPoints.length === 0 && !layer.editState.hasMask);
  imageStage.classList.toggle("active-layer-hidden", !layer.visible);
  if (!layer.visible) hideBrushCursor();
}

function resetMaskUiState(): void {
  activeStroke = null;
  const category = activeCategories()[0] ?? categories[0] ?? defaultCategory();
  maskLayers.reset(category.id, category.name, category.color);
  hideBrushCursor();
  if (isBrushTool(activeTool)) setTool("positive");
  renderMaskPicker();
  updateEditingControls();
}

function descriptor(layer: MaskLayer): { id: string; color: string; visible: boolean } {
  return { id: layer.id, color: layer.color, visible: layer.visible };
}

function renderMaskPicker(): void {
  const active = maskLayers.active();
  const activeCategory = categoryById(active.categoryId);
  const count = maskLayers.all().length;
  activeLayerSwatch.style.background = active.color;
  activeLayerName.textContent = layerDisplayName(active);
  activeLayerName.classList.toggle("hidden-mask", !active.visible);
  maskCount.textContent = `${count} mask${count === 1 ? "" : "s"}`;
  maskLayerName.value = activeCategory?.name ?? active.name;
  maskLayerColor.value = activeCategory?.color ?? active.color;
  newMaskClass.replaceChildren();
  for (const category of activeCategories()) newMaskClass.append(new Option(category.name, String(category.id)));
  if (activeCategories().some((category) => category.id === active.categoryId)) {
    newMaskClass.value = String(active.categoryId);
  }
  maskLayerClass.replaceChildren();
  for (const category of categories.filter((item) => item.active || item.id === active.categoryId)) {
    maskLayerClass.append(new Option(`${category.name}${category.active ? "" : " (removed)"}`, String(category.id)));
  }
  maskLayerClass.value = String(active.categoryId);
  addMaskConfirm.disabled = !imageReady || activeCategories().length === 0;
  archiveClassButton.disabled = !activeCategory?.active;
  deleteMaskButton.disabled = count === 1;
  preventMaskOverlap.checked = preventOverlap;
  addMaskButton.disabled = !imageReady;
  maskLayerList.replaceChildren();
  for (const layer of maskLayers.all()) {
    const row = document.createElement("div");
    row.className = "mask-layer-row";
    row.classList.toggle("active", layer.id === active.id);
    row.setAttribute("role", "option");
    row.setAttribute("aria-selected", String(layer.id === active.id));

    const eye = document.createElement("button");
    eye.type = "button";
    eye.className = "layer-visibility";
    eye.dataset.layerVisible = layer.id;
    eye.setAttribute("aria-pressed", String(layer.visible));
    eye.setAttribute("aria-label", `${layer.visible ? "Hide" : "Show"} ${layer.name}`);
    eye.textContent = layer.visible ? "●" : "○";

    const select = document.createElement("button");
    select.type = "button";
    select.className = "layer-select";
    select.dataset.layerSelect = layer.id;
    const swatch = document.createElement("span");
    swatch.className = "layer-swatch";
    swatch.style.background = layer.color;
    const name = document.createElement("span");
    name.textContent = layerDisplayName(layer);
    select.append(swatch, name);
    row.append(eye, select);
    maskLayerList.append(row);
  }
}

function categoryById(id: number): Category | undefined {
  return categories.find((category) => category.id === id);
}

function layerDisplayName(layer: MaskLayer): string {
  const sameClass = maskLayers.all().filter((item) => item.categoryId === layer.categoryId);
  return `${layer.name} · ${sameClass.indexOf(layer) + 1}`;
}

function activeCategories(): Category[] {
  return categories.filter((category) => category.active);
}

function defaultCategory(): Category {
  return { id: 1, name: "object", supercategory: "phytolith", color: "#4094dc", active: true };
}

function base64ToBytes(value: string): Uint8Array {
  const binary = atob(value);
  const output = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) output[index] = binary.charCodeAt(index);
  return output;
}

function bytesToBase64(value: Uint8Array): string {
  let binary = "";
  const chunk = 0x8000;
  for (let offset = 0; offset < value.length; offset += chunk) {
    binary += String.fromCharCode(...value.subarray(offset, offset + chunk));
  }
  return btoa(binary);
}

async function responseErrorFromFetch(response: Response): Promise<Error> {
  try {
    const body = await response.json() as { detail?: string };
    if (body.detail) return new Error(body.detail);
  } catch { /* Use status fallback. */ }
  return new Error(`Request failed with HTTP ${response.status}.`);
}

function setMaskPickerOpen(open: boolean): void {
  maskPickerMenu.hidden = !open;
  maskPickerTrigger.setAttribute("aria-expanded", String(open));
  maskPicker.classList.toggle("open", open);
  if (open) renderMaskPicker();
}

function renderDataBrowser(): void {
  imageSelect.replaceChildren();
  folderCrumbs.replaceChildren();
  folderTree.replaceChildren();

  if (!dataRoot || !currentFolder) {
    folderTreeValue.textContent = "No data folders";
    folderTreeTrigger.disabled = true;
    setFolderTreeOpen(false);
    imageSelect.append(new Option("No images", ""));
    imageSelect.disabled = true;
    previousButton.disabled = true;
    nextButton.disabled = true;
    navigateFolders.disabled = true;
    navigateImages.disabled = true;
    return;
  }

  folderTreeTrigger.disabled = false;
  folderTreeValue.textContent = currentFolder.path
    ? `Data / ${currentFolder.path}`
    : "Data";
  renderFolderTree();

  if (currentFolder.images.length === 0) {
    imageSelect.append(new Option("No images in this folder", ""));
    imageSelect.disabled = true;
  } else {
    for (const image of currentFolder.images) {
      imageSelect.append(new Option(image.name, image.path));
    }
    imageSelect.value = activeImage?.path ?? "";
    imageSelect.disabled = false;
  }

  const crumbs = folderBreadcrumbs(dataRoot, currentFolder);
  crumbs.forEach((folder, index) => {
    if (index > 0) {
      const separator = document.createElement("span");
      separator.textContent = "/";
      separator.className = "crumb-separator";
      folderCrumbs.append(separator);
    }
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = folder.name;
    button.dataset.folderPath = folder.path;
    button.disabled = folder.path === currentFolder?.path;
    folderCrumbs.append(button);
  });

  const previous =
    navigationMode === "folder"
      ? adjacentFolder(dataRoot, currentFolder, -1)
      : adjacentImage(currentFolder, activeImage, -1);
  const next =
    navigationMode === "folder"
      ? adjacentFolder(dataRoot, currentFolder, 1)
      : adjacentImage(currentFolder, activeImage, 1);
  previousButton.disabled = !previous;
  nextButton.disabled = !next;
  navigateFolders.disabled = false;
  navigateImages.disabled = !activeImage;
  navigateFolders.classList.toggle(
    "active",
    navigationMode === "folder",
  );
  navigateFolders.setAttribute(
    "aria-pressed",
    String(navigationMode === "folder"),
  );
  navigateImages.classList.toggle("active", navigationMode === "image");
  navigateImages.setAttribute(
    "aria-pressed",
    String(navigationMode === "image"),
  );
}

function renderFolderTree(): void {
  folderTree.replaceChildren();
  if (!dataRoot || !currentFolder) return;
  folderTree.append(createFolderTreeNode(dataRoot, currentFolder, 0));
}

function createFolderTreeNode(
  folder: DataFolder,
  selectedFolder: DataFolder,
  depth: number,
): HTMLDivElement {
  const branch = document.createElement("div");
  branch.className = "tree-branch";
  branch.setAttribute("role", "treeitem");
  branch.setAttribute(
    "aria-expanded",
    String(expandedFolderPaths.has(folder.path)),
  );
  branch.setAttribute(
    "aria-selected",
    String(folder.path === selectedFolder.path),
  );

  const row = document.createElement("div");
  row.className = "tree-row";
  row.style.paddingLeft = `${depth * 14 + 6}px`;

  if (folder.folders.length > 0) {
    const expander = document.createElement("button");
    expander.type = "button";
    expander.className = "tree-expander";
    expander.dataset.expandFolder = folder.path;
    expander.textContent = expandedFolderPaths.has(folder.path) ? "▾" : "▸";
    expander.setAttribute(
      "aria-label",
      `${expandedFolderPaths.has(folder.path) ? "Collapse" : "Expand"} ${folder.name}`,
    );
    row.append(expander);
  } else {
    const spacer = document.createElement("span");
    spacer.className = "tree-expander-spacer";
    row.append(spacer);
  }

  const folderButton = document.createElement("button");
  folderButton.type = "button";
  folderButton.className = "tree-folder";
  folderButton.dataset.folderPath = folder.path;
  folderButton.classList.toggle(
    "active",
    folder.path === selectedFolder.path,
  );
  folderButton.innerHTML = `<span aria-hidden="true">▱</span>${escapeHtml(
    folder.name,
  )}`;
  row.append(folderButton);
  branch.append(row);

  if (
    folder.folders.length > 0 &&
    expandedFolderPaths.has(folder.path)
  ) {
    const children = document.createElement("div");
    children.className = "tree-children";
    children.setAttribute("role", "group");
    folder.folders.forEach((child) => {
      children.append(createFolderTreeNode(child, selectedFolder, depth + 1));
    });
    branch.append(children);
  }

  return branch;
}

function expandAncestors(path: string): void {
  expandedFolderPaths.add("");
  const segments = path.split("/").filter(Boolean);
  for (let index = 0; index < segments.length; index += 1) {
    expandedFolderPaths.add(segments.slice(0, index + 1).join("/"));
  }
}

function setFolderTreeOpen(open: boolean): void {
  if (folderTreeTrigger.disabled) open = false;
  folderTreeMenu.hidden = !open;
  folderTreeTrigger.setAttribute("aria-expanded", String(open));
  folderTreePicker.classList.toggle("open", open);
  if (open) renderFolderTree();
}

function setDataImage(image: DataImage): void {
  sourceImage.src = image.url;
  sourceImage.alt = image.name;
  viewerTitle.textContent = image.name;
}

function setStageSize(width: number, height: number): void {
  displayImageSize = { width, height };
  imageStage.style.aspectRatio = `${width} / ${height}`;
  imageFrame.style.width = "100%";

  const frameStyle = getComputedStyle(imageFrame);
  const horizontalPadding =
    Number.parseFloat(frameStyle.paddingLeft) +
    Number.parseFloat(frameStyle.paddingRight);
  const horizontalBorder =
    Number.parseFloat(frameStyle.borderLeftWidth) +
    Number.parseFloat(frameStyle.borderRightWidth);
  const availableWidth = Math.max(
    1,
    imageFrame.clientWidth - horizontalPadding,
  );
  const availableHeight = Math.max(280, window.innerHeight - 250);
  const fitted = fitDimensions(
    width,
    height,
    availableWidth,
    availableHeight,
  );

  imageStage.style.width = `${fitted.width}px`;
  imageStage.style.height = `${fitted.height}px`;
  imageFrame.style.width = `${
    fitted.width + horizontalPadding + horizontalBorder
  }px`;
  if (brushCursor.classList.contains("visible")) resizeBrushCursor();
}

function renderMarkers(): void {
  markerLayer.innerHTML = "";
  const layer = maskLayers.active();
  if (!layer.visible) return;
  layer.pinnedPoints.forEach((point) => {
    markerLayer.append(createMarker(point, false));
  });
  if (hoverPoint) markerLayer.append(createMarker(hoverPoint, true));
}

function updateHoverMarker(point: PointPrompt): void {
  let marker = markerLayer.querySelector<HTMLDivElement>(".marker.hover");
  if (!marker) {
    marker = createMarker(point, true);
    markerLayer.append(marker);
  }
  applyMarker(marker, point);
}

function removeHoverMarker(): void {
  markerLayer.querySelector(".marker.hover")?.remove();
}

function createMarker(point: PointPrompt, hover: boolean): HTMLDivElement {
  const marker = document.createElement("div");
  marker.className = `marker ${point.label === 1 ? "positive" : "negative"} ${hover ? "hover" : "pinned"}`;
  marker.textContent = point.label === 1 ? "+" : "−";
  applyMarker(marker, point);
  return marker;
}

function applyMarker(marker: HTMLDivElement, point: PointPrompt): void {
  marker.classList.toggle("positive", point.label === 1);
  marker.classList.toggle("negative", point.label === 0);
  marker.textContent = point.label === 1 ? "+" : "−";
  marker.style.left = `${point.x * 100}%`;
  marker.style.top = `${point.y * 100}%`;
}

function updatePointControls(): void {
  const layer = maskLayers.active();
  const count = layer.pinnedPoints.length;
  undoButton.disabled = !layer.visible || count === 0;
  clearButton.disabled = !layer.visible || (count === 0 && !layer.editState.hasMask);
  pointCount.textContent = `${count} pinned`;
}

function setStatus(
  state: "loading" | "ready" | "error",
  message: string,
): void {
  statusChip.className = `status-chip ${state}`;
  statusLabel.textContent = message;
}

function showStageMessage(title: string, detail: string): void {
  stageMessage.classList.remove("hidden", "error");
  stageMessage.innerHTML = `
    <span class="loader"></span>
    <strong>${title}</strong>
    <small>${detail}</small>
  `;
}

function showNotice(title: string, detail: string): void {
  stageMessage.classList.remove("hidden", "error");
  stageMessage.innerHTML = `
    <span class="notice-symbol">⌁</span>
    <strong>${escapeHtml(title)}</strong>
    <small>${escapeHtml(detail)}</small>
  `;
}

function countImages(folder: DataFolder): number {
  return (
    folder.images.length +
    folder.folders.reduce(
      (total, child) => total + countImages(child),
      0,
    )
  );
}

function showFatal(message: string, label = "Runtime error"): void {
  imageReady = false;
  imageStage.classList.add("disabled");
  setStatus("error", label);
  stageMessage.classList.remove("hidden");
  stageMessage.classList.add("error");
  stageMessage.innerHTML = `
    <span class="error-symbol">!</span>
    <strong>Unable to start the demo</strong>
    <small>${escapeHtml(message)}</small>
  `;
  progressTrack.classList.remove("visible");
}

function formatDuration(milliseconds: number): string {
  if (milliseconds >= 1000) return `${(milliseconds / 1000).toFixed(1)} s`;
  return `${Math.round(milliseconds)} ms`;
}

function getElement<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Missing required element #${id}.`);
  return element as T;
}

function escapeHtml(value: string): string {
  return value.replace(
    /[&<>"']/g,
    (character) =>
      ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;",
      })[character] ?? character,
  );
}
