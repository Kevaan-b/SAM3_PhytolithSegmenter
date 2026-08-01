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
      <div class="header-controls">
        <div class="model-pill"><span class="model-dot"></span>SAM3 Q4 · H100 CUDA</div>
        <div class="status-chip loading" id="status-chip" role="status" aria-live="polite"><span class="status-light"></span><span id="status-label">Connecting to H100…</span></div>
        <div class="save-controls"><button class="autosave-button" id="autosave-toggle" type="button" aria-pressed="true" title="Autosave is on">Auto</button><button id="save-annotations" type="button" disabled>Saved</button></div>
      </div>
    </header>

    <main class="workspace">
      <aside class="control-panel">
        <div class="sidebar-tabs" role="tablist" aria-label="Workspace controls">
          <button class="sidebar-tab" id="setup-tab" type="button" role="tab" aria-controls="setup-panel" aria-selected="false" tabindex="-1">Setup</button>
          <button class="sidebar-tab active" id="masking-tab" type="button" role="tab" aria-controls="masking-panel" aria-selected="true">Masking</button>
          <button class="sidebar-tab" id="statistics-tab" type="button" role="tab" aria-controls="statistics-panel" aria-selected="false" tabindex="-1">Statistics</button>
        </div>

        <div class="sidebar-panel" id="setup-panel" role="tabpanel" aria-labelledby="setup-tab" hidden>
          <section class="control-section data-section">
            <div class="section-heading"><span>01</span><h2>Browse data</h2></div>
            <div class="data-browser">
              <nav class="folder-crumbs" id="folder-crumbs" aria-label="Data folder path"></nav>
              <span class="browser-label" id="folder-tree-label">Folder</span>
              <div class="folder-tree-picker" id="folder-tree-picker">
                <button class="folder-tree-trigger" id="folder-tree-trigger" type="button" aria-labelledby="folder-tree-label folder-tree-value" aria-haspopup="tree" aria-expanded="false" disabled><span id="folder-tree-value">Scanning data/…</span><span aria-hidden="true">⌄</span></button>
                <div class="folder-tree-menu" id="folder-tree-menu" hidden><div class="folder-tree" id="folder-tree" role="tree"></div></div>
              </div>
              <button class="nav-button next-folder-button" id="next-folder-button" type="button" disabled>Next folder →</button>
              <button class="refresh-button" id="refresh-data" type="button">↻ Rescan data folder</button>
              <p class="data-summary" id="data-summary">Looking for images under data/…</p>
            </div>
          </section>

          <section class="control-section class-setup-section">
            <div class="section-heading"><span>02</span><h2>Class definitions</h2></div>
            <p class="panel-hint">Names and colors apply everywhere this class is used.</p>
            <input class="class-search" id="setup-class-search" type="search" placeholder="Search classes" aria-label="Search class definitions" />
            <div class="class-list setup-class-list" id="setup-class-list" role="listbox" aria-label="Class definitions"></div>
            <div class="mask-layer-editor class-definition-editor">
              <label for="mask-layer-name">Name</label><input id="mask-layer-name" maxlength="80" />
              <label for="mask-layer-color">Color</label><input id="mask-layer-color" type="color" />
              <button class="archive-class-button" id="archive-class" type="button">Remove class</button>
            </div>
            <form class="new-class-form" id="new-class-form">
              <label for="new-class-name">Add class</label><input id="new-class-name" maxlength="80" placeholder="Class name" /><button id="new-class" type="submit">Add</button>
            </form>
          </section>
        </div>

        <div class="sidebar-panel" id="masking-panel" role="tabpanel" aria-labelledby="masking-tab">
          <section class="control-section image-navigation-section">
            <div class="section-heading"><span>01</span><h2>Image</h2></div>
            <label class="browser-label" for="image-select">Current image</label>
            <select class="browser-select" id="image-select" disabled><option>No images</option></select>
            <div class="image-navigation">
              <button class="nav-button" id="previous-image-button" type="button" disabled>← Previous image</button>
              <button class="nav-button" id="next-image-button" type="button" disabled>Next image →</button>
            </div>
          </section>
          <section class="control-section masking-class-section">
            <div class="section-heading"><span>02</span><h2>Class</h2></div>
            <p class="panel-hint">Choose the class for the active mask and the next new mask.</p>
            <input class="class-search" id="masking-class-search" type="search" placeholder="Search classes" aria-label="Search mask classes" />
            <div class="class-list" id="new-mask-class" role="listbox" aria-label="Mask class"></div>
          </section>
          <section class="control-section masks-section">
            <div class="section-heading"><span>03</span><h2>Masks</h2><span class="mask-count" id="mask-count">1 mask</span></div>
            <div class="mask-picker" id="mask-picker">
              <div class="mask-picker-toolbar"><button class="add-mask-button" id="add-mask" type="button">+ New mask</button></div>
              <div class="mask-layer-list" id="mask-layer-list" role="listbox" aria-label="Mask layers"></div>
              <div class="mask-actions"><button class="text-button danger" id="delete-mask" type="button">Delete mask</button></div>
              <label class="overlap-toggle" for="prevent-mask-overlap"><span><strong>Prevent overlaps</strong><small>Clip the latest mask against all others</small></span><input id="prevent-mask-overlap" type="checkbox" /></label>
            </div>
          </section>
        </div>

        <div class="sidebar-panel statistics-panel" id="statistics-panel" role="tabpanel" aria-labelledby="statistics-tab" hidden>
          <section class="control-section">
            <div class="section-heading"><span>01</span><h2>Annotation statistics</h2></div>
            <p class="panel-hint">Saved instances across the annotations folder. Refreshes whenever you open this tab.</p>
            <div class="statistics-total" id="statistics-total" aria-live="polite">Loading saved annotations…</div>
            <input class="class-search" id="statistics-class-search" type="search" placeholder="Search classes" aria-label="Search statistics classes" />
            <div class="statistics-list" id="statistics-list" aria-live="polite"></div>
          </section>
        </div>
      </aside>

      <section class="viewer-panel" aria-label="Interactive segmentation viewer">
        <div class="viewer-topline">
          <div>
            <div class="eyebrow">Live canvas</div>
            <h2 id="viewer-title">No image selected</h2>
          </div>
        </div>

        <div class="statistics-viewer" id="statistics-viewer" hidden>
          <p class="statistics-viewer-hint">Saved masks are shown as color overlays. Select an image to open it in Masking.</p>
          <div class="statistics-pagination" id="statistics-pagination" hidden>
            <span id="statistics-image-counter">Images 0 of 0</span>
            <div class="statistics-page-controls">
              <button id="statistics-previous-page" type="button">← Previous</button>
              <span id="statistics-page-counter">Page 0 of 0</span>
              <button id="statistics-next-page" type="button">Next →</button>
            </div>
          </div>
          <div class="statistics-preview-grid" id="statistics-preview-grid"></div>
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

        <div class="viewer-footer" id="viewer-footer">
          <p id="interaction-hint"><span class="cursor-symbol">⌖</span> Hover to preview · Click to pin</p>
          <dl class="metrics">
            <div><dt>Encoder</dt><dd id="encode-metric">—</dd></div>
            <div><dt>Last decode</dt><dd id="decode-metric">—</dd></div>
          </dl>
        </div>
      </section>

      <aside class="refine-panel" aria-label="Point and mask refinement controls">
        <section class="control-section point-type-section">
          <div class="section-heading"><span>01</span><h2>Point type</h2></div>
          <div class="tool-switch" role="group" aria-label="Point type"><button class="tool-button active" id="positive-tool" type="button" aria-pressed="true"><span class="tool-icon positive">+</span>Positive</button><button class="tool-button" id="negative-tool" type="button" aria-pressed="false"><span class="tool-icon negative">−</span>Negative</button></div>
          <p class="tool-hint" id="tool-hint">Positive points include a region.</p>
        </section>
        <div class="point-actions"><button class="text-button" id="undo-button" type="button" disabled><span aria-hidden="true">↶</span> Undo</button><button class="text-button" id="clear-button" type="button" disabled><span aria-hidden="true">×</span> Clear all</button><span class="point-count" id="point-count">0 pinned</span></div>
        <section class="control-section mask-edit-section">
          <div class="section-heading"><span>02</span><h2>Refine mask</h2></div>
          <div class="tool-switch" role="group" aria-label="Mask brush"><button class="tool-button" id="marker-tool" type="button" aria-pressed="false" disabled><span class="tool-icon marker-icon">●</span>Marker</button><button class="tool-button" id="eraser-tool" type="button" aria-pressed="false" disabled><span class="tool-icon eraser-icon">○</span>Eraser</button></div>
          <div class="brush-controls"><label for="marker-size"><span>Marker size</span><output id="marker-size-value">20 px</output></label><input id="marker-size" type="range" min="2" max="200" step="2" value="20" /><label for="eraser-size"><span>Eraser size</span><output id="eraser-size-value">32 px</output></label><input id="eraser-size" type="range" min="2" max="200" step="2" value="32" /></div>
          <div class="mask-actions"><button class="text-button" id="invert-mask" type="button" aria-pressed="false" disabled>◐ Invert mask</button><button class="text-button" id="undo-edit" type="button" disabled>↶ Undo edit</button><button class="text-button" id="reset-edits" type="button" disabled>× Reset edits</button></div>
        </section>
      </aside>
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
const nextFolderButton = getElement<HTMLButtonElement>("next-folder-button");
const previousImageButton = getElement<HTMLButtonElement>("previous-image-button");
const nextImageButton = getElement<HTMLButtonElement>("next-image-button");
const refreshDataButton = getElement<HTMLButtonElement>("refresh-data");
const dataSummary = getElement<HTMLParagraphElement>("data-summary");
const imageStage = getElement<HTMLDivElement>("image-stage");
const sourceImage = getElement<HTMLImageElement>("source-image");
const overlay = getElement<HTMLCanvasElement>("mask-overlay");
const markerLayer = getElement<HTMLDivElement>("marker-layer");
const setupTab = getElement<HTMLButtonElement>("setup-tab");
const maskingTab = getElement<HTMLButtonElement>("masking-tab");
const statisticsTab = getElement<HTMLButtonElement>("statistics-tab");
const setupPanel = getElement<HTMLDivElement>("setup-panel");
const maskingPanel = getElement<HTMLDivElement>("masking-panel");
const statisticsPanel = getElement<HTMLDivElement>("statistics-panel");
const statisticsTotal = getElement<HTMLDivElement>("statistics-total");
const statisticsList = getElement<HTMLDivElement>("statistics-list");
const statisticsClassSearch = getElement<HTMLInputElement>("statistics-class-search");
const setupClassList = getElement<HTMLDivElement>("setup-class-list");
const setupClassSearch = getElement<HTMLInputElement>("setup-class-search");
const maskLayerList = getElement<HTMLDivElement>("mask-layer-list");
const maskCount = getElement<HTMLSpanElement>("mask-count");
const addMaskButton = getElement<HTMLButtonElement>("add-mask");
const newMaskClass = getElement<HTMLDivElement>("new-mask-class");
const maskingClassSearch = getElement<HTMLInputElement>("masking-class-search");
const newClassForm = getElement<HTMLFormElement>("new-class-form");
const newClassName = getElement<HTMLInputElement>("new-class-name");
const maskLayerName = getElement<HTMLInputElement>("mask-layer-name");
const maskLayerColor = getElement<HTMLInputElement>("mask-layer-color");
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
const statisticsViewer = getElement<HTMLDivElement>("statistics-viewer");
const statisticsPreviewGrid = getElement<HTMLDivElement>("statistics-preview-grid");
const statisticsPagination = getElement<HTMLDivElement>("statistics-pagination");
const statisticsImageCounter = getElement<HTMLSpanElement>("statistics-image-counter");
const statisticsPageCounter = getElement<HTMLSpanElement>("statistics-page-counter");
const statisticsPreviousPage = getElement<HTMLButtonElement>("statistics-previous-page");
const statisticsNextPage = getElement<HTMLButtonElement>("statistics-next-page");
const viewerFooter = getElement<HTMLElement>("viewer-footer");
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
const autosaveToggle = getElement<HTMLButtonElement>("autosave-toggle");

interface Category { id: number; name: string; supercategory: string; color: string; active: boolean }
interface StatisticsPreview {
  imageId: string;
  fileName: string;
  annotationCount: number;
  savedAt: string;
  previewUrl: string;
}

interface ClassStatistics {
  totalAnnotations: number;
  classes: Array<Category & { annotationCount: number }>;
}

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
let selectedClassId = 1;
let classUsageTick = 0;
let lastTouchedClassId: number | null = null;
const classLastUsed = new Map<number, number>();
let activeSidebarTab: "setup" | "masking" | "statistics" = "masking";
let statisticsSnapshot: ClassStatistics | null = null;
let selectedStatisticsClassId: number | null = null;
let statisticsController: AbortController | null = null;
let previewController: AbortController | null = null;
let statisticsPage = 1;
let statisticsTotalPages = 0;
let preventOverlap = false;
let dirty = false;
let annotationRevision = 0;
let saving = false;
let lastSaveError: string | null = null;
let saveTimer = 0;
let autosaveEnabled = true;
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
setSidebarTab(activeSidebarTab);

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
        const folder = message.activeFolder;
        dataSummary.textContent = folder && folder.total > 0
          ? `${folder.ready}/${folder.total} ready in ${folder.path || "Data"} · ${message.gpuResident} on H100 · ${message.queueDepth} queued${message.backgroundPaused ? " · paused while annotating" : ""}`
          : `${message.ready} embeddings cached · ${message.gpuResident} on H100`;

        if (folder && folder.total > 0 && folder.ready < folder.total) {
          progressTrack.classList.add("visible");
          progressBar.style.width = String((folder.ready / folder.total) * 100) + "%";
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
        setStatus("ready", "Ready");
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
        setStatus("ready", "Ready");
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
      void refreshStatistics(activeSidebarTab === "statistics");
      return true;
    } catch (error) {
      lastSaveError = error instanceof Error ? error.message : "Unknown save error";
      return false;
    } finally {
      saving = false;
      updateSaveState();
      if (autosaveEnabled && dirty && !lastSaveError) {
        window.clearTimeout(saveTimer);
        saveTimer = window.setTimeout(() => void saveAnnotations(), 400);
      }
    }
  }

  function markDirty(): void {
    dirty = true;
    annotationRevision += 1;
    updateSaveState();
    if (autosaveEnabled) {
      window.clearTimeout(saveTimer);
      saveTimer = window.setTimeout(() => void saveAnnotations(), 1200);
    }
  }

  function updateSaveState(): void {
    saveAnnotationsButton.disabled = !activeImage || !imageReady || saving || !dirty;
    saveAnnotationsButton.textContent = saving
      ? "Saving…"
      : lastSaveError
        ? "Save failed"
        : dirty
          ? "Save"
          : "Saved";
    autosaveToggle.disabled = !activeImage || !imageReady;
    autosaveToggle.setAttribute("aria-pressed", String(autosaveEnabled));
    autosaveToggle.title = autosaveEnabled ? "Autosave is on" : "Autosave is off";
  }

  saveAnnotationsButton.addEventListener("click", () => void saveAnnotations());
  autosaveToggle.addEventListener("click", () => {
    autosaveEnabled = !autosaveEnabled;
    window.clearTimeout(saveTimer);
    if (autosaveEnabled && dirty && !saving && !lastSaveError) {
      saveTimer = window.setTimeout(() => void saveAnnotations(), 400);
    }
    updateSaveState();
  });
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
    postCacheInteraction(true);
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
    if (!activeStroke) postCacheInteraction(false);
    latestPointer = null;
    hoverPoint = null;
    removeHoverMarker();
    hideBrushCursor();
    if (isBrushTool(activeTool)) return;
    const layer = maskLayers.active();
    layer.stateRevision += 1;
    layer.lastPromptKey = promptKey(layer.pinnedPoints);
    postWorker({
      type: "cancel-preview",
      imageRevision,
      layerId: layer.id,
      stateRevision: layer.stateRevision,
    });
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
    if (!pointerInside) postCacheInteraction(false);
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
    void prioritizeCache(folder.path, image?.id);

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
    if (currentFolder) void prioritizeCache(currentFolder.path, image.id);
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

  nextFolderButton.addEventListener("click", () => {
    navigationMode = "folder";
    moveSelection(1);
  });
  previousImageButton.addEventListener("click", () => {
    navigationMode = "image";
    moveSelection(-1);
  });
  nextImageButton.addEventListener("click", () => {
    navigationMode = "image";
    moveSelection(1);
  });
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
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") setFolderTreeOpen(false);
  });

  const sidebarTabs = [setupTab, maskingTab, statisticsTab];
  const sidebarTabNames: Array<"setup" | "masking" | "statistics"> = ["setup", "masking", "statistics"];
  setupTab.addEventListener("click", () => setSidebarTab("setup"));
  maskingTab.addEventListener("click", () => setSidebarTab("masking"));
  statisticsTab.addEventListener("click", () => setSidebarTab("statistics"));
  sidebarTabs.forEach((tab, index) => tab.addEventListener("keydown", (event) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const direction = event.key === "ArrowRight" ? 1 : -1;
    const nextIndex = (index + direction + sidebarTabs.length) % sidebarTabs.length;
    const nextTab = sidebarTabs[nextIndex];
    const nextName = sidebarTabNames[nextIndex];
    if (!nextTab || !nextName) return;
    setSidebarTab(nextName);
    nextTab.focus();
  }));
  setupClassList.addEventListener("click", (event) => {
    const button = (event.target as HTMLElement).closest<HTMLButtonElement>("[data-class-id]");
    if (!button) return;
    selectedClassId = Number(button.dataset.classId);
    renderMaskPicker();
  });
  setupClassSearch.addEventListener("input", renderMaskPicker);
  maskingClassSearch.addEventListener("input", renderMaskPicker);
  statisticsClassSearch.addEventListener("input", () => {
    if (statisticsSnapshot) renderStatisticsRows(statisticsSnapshot.classes);
  });
  statisticsList.addEventListener("click", (event) => {
    const button = (event.target as HTMLElement).closest<HTMLButtonElement>("[data-statistics-class]");
    if (!button || !statisticsSnapshot) return;
    selectedStatisticsClassId = Number(button.dataset.statisticsClass);
    statisticsPage = 1;
    renderStatisticsRows(statisticsSnapshot.classes);
    void refreshStatisticsPreviews(selectedStatisticsClassId);
  });
  statisticsPreviousPage.addEventListener("click", () => {
    if (statisticsPage <= 1) return;
    statisticsPage -= 1;
    void refreshStatisticsPreviews(selectedStatisticsClassId, statisticsPage);
  });
  statisticsNextPage.addEventListener("click", () => {
    if (statisticsPage >= statisticsTotalPages) return;
    statisticsPage += 1;
    void refreshStatisticsPreviews(selectedStatisticsClassId, statisticsPage);
  });
  statisticsPreviewGrid.addEventListener("click", (event) => {
    const card = (event.target as HTMLElement).closest<HTMLElement>("[data-statistics-image]");
    const imageId = card?.dataset.statisticsImage;
    if (!imageId) return;
    void (async () => {
      const match = findStatisticsImage(imageId);
      if (!match) {
        showNotice("Image unavailable", "Rescan data/ if this image was recently moved.");
        return;
      }
      if (dirty && !(await saveAnnotations())) return;
      currentFolder = match.folder;
      expandAncestors(match.folder.path);
      void prioritizeCache(match.folder.path, match.image.id);
      await selectImageAfterSave(match.image);
      setSidebarTab("masking");
    })();
  });

  addMaskButton.addEventListener("click", () => {
    if (!imageReady) return;
    const category = categoryById(Number(newMaskClass.dataset.selected));
    if (!category) return;
    addMaskForCategory(category);
  });
  newMaskClass.addEventListener("click", (event) => {
    const button = (event.target as HTMLElement).closest<HTMLButtonElement>("[data-class-id]");
    if (!button) return;
    const category = categoryById(Number(button.dataset.classId));
    if (!category) return;
    setActiveLayerCategory(category);
  });
  newClassForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const name = newClassName.value.trim();
    if (name) { newClassName.value = ""; void addCategory(name); }
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
    touchClass(layer.categoryId);
    postWorker({ type: "activate-layer", imageRevision, layerId: layer.id });
    refreshActiveLayerUi();
  });
  maskLayerName.addEventListener("change", () => {
    void updateCategory(selectedClassId, { name: maskLayerName.value });
  });
  maskLayerColor.addEventListener("change", () => {
    void updateCategory(selectedClassId, { color: maskLayerColor.value });
  });
  archiveClassButton.addEventListener("click", () => {
    const category = categoryById(selectedClassId);
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
    promoteClass(category.id);
    selectedClassId = category.id;
    renderMaskPicker();
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

  function addMaskForCategory(category: Category): void {
    touchClass(category.id);
    cancelTransientInteraction();
    const layer = maskLayers.add(category.id, category.name, category.color);
    postWorker({ type: "create-layer", imageRevision, layer: descriptor(layer) });
    postWorker({ type: "activate-layer", imageRevision, layerId: layer.id });
    refreshActiveLayerUi();
    markDirty();
  }

  function setActiveLayerCategory(category: Category): void {
    touchClass(category.id);
    selectedClassId = category.id;
    const layer = maskLayers.active();
    maskLayers.setCategory(layer.id, category.id, category.name, category.color);
    postWorker({ type: "update-layer", imageRevision, layerId: layer.id, color: category.color });
    renderMaskPicker();
    markDirty();
  }
}

function postCacheInteraction(active: boolean): void {
  void fetch("/api/cache/interaction", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ active }), keepalive: true }).catch(() => undefined);
}

async function prioritizeCache(folderPath: string, imageId?: string): Promise<void> {
  try {
    await fetch("/api/cache/prioritize", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ folderPath, imageId }) });
  } catch { /* Image preparation will retry when the service becomes ready. */ }
}

function renderStatisticsRows(classes: Array<Category & { annotationCount: number }>): void {
  statisticsList.replaceChildren();
  const query = statisticsClassSearch.value.trim().toLocaleLowerCase();
  for (const category of classes.filter((item) => item.name.toLocaleLowerCase().includes(query))) {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "statistics-row";
    row.dataset.statisticsClass = String(category.id);
    row.setAttribute("role", "option");
    row.setAttribute("aria-selected", String(category.id === selectedStatisticsClassId));
    const swatch = document.createElement("span");
    swatch.className = "layer-swatch";
    swatch.style.background = category.color;
    const name = document.createElement("strong");
    name.textContent = category.name;
    const count = document.createElement("span");
    count.className = "statistics-count";
    count.textContent = String(category.annotationCount) + " " + (category.annotationCount === 1 ? "annotation" : "annotations");
    row.append(swatch, name, count);
    statisticsList.append(row);
  }
  if (classes.length === 0) statisticsList.textContent = "No classes have been defined yet.";
}

function renderStatisticsPreviews(previews: StatisticsPreview[]): void {
  statisticsPreviewGrid.replaceChildren();
  for (const preview of previews.slice(0, 8)) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "statistics-preview-card";
    card.dataset.statisticsImage = preview.imageId;
    card.title = `Open ${preview.fileName} in Masking`;
    const image = document.createElement("img");
    image.src = preview.previewUrl;
    image.loading = "lazy";
    image.decoding = "async";
    image.alt = preview.fileName + " with selected class overlay";
    const caption = document.createElement("span");
    caption.className = "statistics-preview-caption";
    caption.textContent = preview.fileName;
    card.append(image, caption);
    statisticsPreviewGrid.append(card);
  }
  if (previews.length === 0) statisticsPreviewGrid.textContent = "No saved previews for this class.";
}

function findStatisticsImage(imageId: string): { folder: DataFolder; image: DataImage } | null {
  if (!dataRoot) return null;
  for (const folder of flattenFolders(dataRoot)) {
    const image = folder.images.find((candidate) => candidate.id === imageId);
    if (image) return { folder, image };
  }
  return null;
}

function renderStatisticsPagination(result: { page: number; pageSize: number; totalImages: number; totalPages: number; previews: StatisticsPreview[] }): void {
  const start = result.totalImages === 0 ? 0 : (result.page - 1) * result.pageSize + 1;
  const end = result.totalImages === 0 ? 0 : start + result.previews.length - 1;
  statisticsImageCounter.textContent = result.totalImages === 1
    ? "Image 1 of 1"
    : `Images ${start}–${end} of ${result.totalImages}`;
  const shownPage = result.totalPages === 0 ? 0 : result.page;
  const remaining = Math.max(0, result.totalPages - shownPage);
  statisticsPageCounter.textContent = `Page ${shownPage} of ${result.totalPages} · ${remaining} remaining`;
  statisticsPreviousPage.disabled = shownPage <= 1;
  statisticsNextPage.disabled = shownPage === 0 || shownPage >= result.totalPages;
  statisticsPagination.hidden = false;
}

async function refreshStatisticsPreviews(categoryId: number | null, requestedPage = 1): Promise<void> {
  previewController?.abort();
  statisticsPreviewGrid.replaceChildren();
  statisticsPagination.hidden = true;
  if (categoryId === null || activeSidebarTab !== "statistics") return;
  statisticsPreviewGrid.textContent = "Loading up to 8 previews…";
  const controller = new AbortController();
  previewController = controller;
  try {
    const response = await fetch(`/api/statistics/classes/${categoryId}/previews?page=${requestedPage}`, { signal: controller.signal });
    if (!response.ok) throw await responseErrorFromFetch(response);
    const result = await response.json() as { previews: StatisticsPreview[]; page: number; pageSize: number; totalImages: number; totalPages: number };
    if (previewController === controller && selectedStatisticsClassId === categoryId) {
      statisticsPage = result.page;
      statisticsTotalPages = result.totalPages;
      renderStatisticsPreviews(result.previews);
      renderStatisticsPagination(result);
    }
  } catch (error) {
    if (!(error instanceof DOMException && error.name === "AbortError")) statisticsPreviewGrid.textContent = error instanceof Error ? error.message : String(error);
  }
}

async function refreshStatistics(render = true): Promise<void> {
  statisticsController?.abort();
  const controller = new AbortController();
  statisticsController = controller;
  if (render) {
    statisticsTotal.textContent = "Refreshing saved annotations…";
    statisticsList.replaceChildren();
  }
  try {
    const response = await fetch("/api/statistics", { cache: "no-store", signal: controller.signal });
    if (!response.ok) throw await responseErrorFromFetch(response);
    const statistics = await response.json() as ClassStatistics;
    const classes = [...statistics.classes].sort((left, right) =>
      right.annotationCount - left.annotationCount || left.name.localeCompare(right.name));
    statisticsSnapshot = { ...statistics, classes };
    if (!render || activeSidebarTab !== "statistics" || statisticsController !== controller) return;
    statisticsTotal.textContent = statistics.totalAnnotations === 1
      ? "1 saved annotation"
      : String(statistics.totalAnnotations) + " saved annotations";
    const retainedClass = classes.some((item) => item.id === selectedStatisticsClassId);
    selectedStatisticsClassId = retainedClass ? selectedStatisticsClassId : classes[0]?.id ?? null;
    if (!retainedClass) statisticsPage = 1;
    renderStatisticsRows(classes);
    void refreshStatisticsPreviews(selectedStatisticsClassId, statisticsPage);
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") return;
    if (render) {
      statisticsTotal.textContent = "Statistics unavailable";
      statisticsList.textContent = error instanceof Error ? error.message : String(error);
    }
  }
}

function setStatisticsViewer(visible: boolean): void {
  statisticsViewer.hidden = !visible;
  imageFrame.hidden = visible;
  viewerFooter.hidden = visible;
  progressTrack.hidden = visible;
  if (visible) {
    viewerTitle.textContent = "Annotation previews";
  } else {
    viewerTitle.textContent = activeImage?.name ?? currentFolder?.name ?? "No image selected";
  }
}

function setSidebarTab(tab: "setup" | "masking" | "statistics"): void {
  activeSidebarTab = tab;
  const tabs: Array<[HTMLButtonElement, "setup" | "masking" | "statistics"]> = [
    [setupTab, "setup"], [maskingTab, "masking"], [statisticsTab, "statistics"],
  ];
  const panels: Array<[HTMLDivElement, "setup" | "masking" | "statistics"]> = [
    [setupPanel, "setup"], [maskingPanel, "masking"], [statisticsPanel, "statistics"],
  ];
  tabs.forEach(([button, value]) => {
    const selected = value === tab;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-selected", String(selected));
    button.tabIndex = selected ? 0 : -1;
  });
  panels.forEach(([panel, value]) => { panel.hidden = value !== tab; });
  setStatisticsViewer(tab === "statistics");
  if (tab === "statistics") { postCacheInteraction(false); void refreshStatistics(); }
  else { statisticsController?.abort(); previewController?.abort(); statisticsPreviewGrid.replaceChildren(); statisticsPagination.hidden = true; }
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
  maskCount.textContent = `${count} mask${count === 1 ? "" : "s"}`;
  const selectedCategory = categoryById(selectedClassId) ?? activeCategory ?? defaultCategory();
  selectedClassId = selectedCategory.id;
  maskLayerName.value = selectedCategory.name;
  maskLayerColor.value = selectedCategory.color;
  renderClassList(newMaskClass, filterClasses(activeCategories(), maskingClassSearch.value), active.categoryId);
  newMaskClass.dataset.selected = String(active.categoryId);
  renderClassList(
    setupClassList,
    filterClasses(categories.filter((category) => category.active || category.id === selectedClassId), setupClassSearch.value),
    selectedClassId,
  );
  archiveClassButton.disabled = !selectedCategory.active;
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

function touchClass(categoryId: number): void {
  if (lastTouchedClassId === categoryId) promoteClass(categoryId);
  lastTouchedClassId = categoryId;
}

function promoteClass(categoryId: number): void {
  classUsageTick += 1;
  classLastUsed.set(categoryId, classUsageTick);
}

function filterClasses(values: Category[], query: string): Category[] {
  const normalized = query.trim().toLocaleLowerCase();
  return normalized ? values.filter((category) => category.name.toLocaleLowerCase().includes(normalized)) : values;
}

function orderClasses(values: Category[]): Category[] {
  return values
    .map((category, index) => ({ category, index }))
    .sort((left, right) => {
      const usage = (classLastUsed.get(right.category.id) ?? 0) - (classLastUsed.get(left.category.id) ?? 0);
      return usage || left.index - right.index;
    })
    .map(({ category }) => category);
}

function renderClassList(
  target: HTMLDivElement,
  values: Category[],
  selectedId: number,
): void {
  target.replaceChildren();
  for (const category of orderClasses(values)) {
    const option = document.createElement("button");
    option.type = "button";
    option.className = "class-option";
    option.dataset.classId = String(category.id);
    option.setAttribute("role", "option");
    option.setAttribute("aria-selected", String(category.id === selectedId));
    option.setAttribute("aria-label", category.name + (category.active ? "" : " (removed)"));
    const swatch = document.createElement("span");
    swatch.className = "layer-swatch";
    swatch.style.background = category.color;
    const name = document.createElement("span");
    name.textContent = category.name + (category.active ? "" : " (removed)");
    option.append(swatch, name);
    target.append(option);
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
    nextFolderButton.disabled = true;
    previousImageButton.disabled = true;
    nextImageButton.disabled = true;
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

  nextFolderButton.disabled = !adjacentFolder(dataRoot, currentFolder, 1);
  previousImageButton.disabled = !adjacentImage(currentFolder, activeImage, -1);
  nextImageButton.disabled = !adjacentImage(currentFolder, activeImage, 1);
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
