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
  MainToWorkerMessage,
  PointLabel,
  PointPrompt,
  WorkerToMainMessage,
} from "./protocol";

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

        <section class="control-section">
          <div class="section-heading">
            <span>02</span>
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
        </div>

        <div class="progress-track" id="progress-track" aria-hidden="true">
          <span id="progress-bar"></span>
        </div>

        <div class="image-frame" id="image-frame">
          <div class="image-stage disabled" id="image-stage">
            <img id="source-image" alt="Selected data image" draggable="false" />
            <canvas id="mask-overlay" aria-hidden="true"></canvas>
            <div class="marker-layer" id="marker-layer" aria-hidden="true"></div>
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
          <p><span class="cursor-symbol">⌖</span> Hover to preview · Click to pin</p>
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
const positiveTool = getElement<HTMLButtonElement>("positive-tool");
const negativeTool = getElement<HTMLButtonElement>("negative-tool");
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

let dataRoot: DataFolder | null = null;
let currentFolder: DataFolder | null = null;
let activeImage: DataImage | null = null;
let navigationMode: NavigationMode = "folder";
const expandedFolderPaths = new Set<string>([""]);
let activeTool: PointLabel = 1;
let pinnedPoints: PointPrompt[] = [];
let hoverPoint: PointPrompt | null = null;
let pointerInside = false;
let modelReady = false;
let imageReady = false;
let imageRevision = 0;
let stateRevision = 0;
let animationFrame = 0;
let latestPointer: { clientX: number; clientY: number } | null = null;
let lastPromptKey: string | null = null;
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
updatePointControls();

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
  void refreshData();

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

  function loadActiveImage(): void {
    if (!modelReady || !activeImage) return;
    imageReady = false;
    imageStage.classList.add("disabled");
    setStatus("loading", "Encoding image…");
    showStageMessage(
      "Encoding this image once",
      "The cached embeddings make every point prompt much faster.",
    );
    postWorker({
      type: "load-image",
      imageRevision,
      imageId: activeImage.id,
      url: activeImage.url,
    });
  }

  function submitPrompts(points: PointPrompt[]): void {
    if (!imageReady) return;
    const key = promptKey(points);
    if (key === lastPromptKey) return;
    lastPromptKey = key;
    stateRevision += 1;

    if (points.length === 0) {
      postWorker({
        type: "clear",
        imageRevision,
        stateRevision,
      });
      return;
    }

    postWorker({
      type: "decode",
      imageRevision,
      stateRevision,
      points,
    });
  }

  imageStage.addEventListener("pointerenter", (event) => {
    if (!imageReady) return;
    pointerInside = true;
    queuePointer(event.clientX, event.clientY);
  });

  imageStage.addEventListener("pointermove", (event) => {
    if (!imageReady) return;
    pointerInside = true;
    queuePointer(event.clientX, event.clientY);
  });

  imageStage.addEventListener("pointerleave", () => {
    pointerInside = false;
    latestPointer = null;
    hoverPoint = null;
    removeHoverMarker();
    lastPromptKey = null;
    submitPrompts(pinnedPoints);
  });

  imageStage.addEventListener("click", (event) => {
    if (!imageReady || !pointerInside) return;
    const normalized = normalizePointer(
      event.clientX,
      event.clientY,
      imageStage.getBoundingClientRect(),
    );
    pinnedPoints.push({ ...normalized, label: activeTool });
    hoverPoint = null;
    latestPointer = null;
    lastPromptKey = null;
    renderMarkers();
    updatePointControls();
    submitPrompts(pinnedPoints);
  });

  function queuePointer(clientX: number, clientY: number): void {
    latestPointer = { clientX, clientY };
    if (animationFrame !== 0) return;

    animationFrame = requestAnimationFrame(() => {
      animationFrame = 0;
      if (!latestPointer || !pointerInside || !imageReady) return;
      const normalized = normalizePointer(
        latestPointer.clientX,
        latestPointer.clientY,
        imageStage.getBoundingClientRect(),
      );
      hoverPoint = { ...normalized, label: activeTool };
      updateHoverMarker(hoverPoint);
      submitPrompts([...pinnedPoints, hoverPoint]);
    });
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
    if (activeImage?.path === image.path) {
      renderDataBrowser();
      return;
    }

    activeImage = image;
    imageRevision += 1;
    stateRevision = 0;
    pinnedPoints = clearPoints();
    hoverPoint = null;
    pointerInside = false;
    latestPointer = null;
    lastPromptKey = null;
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
    stateRevision = 0;
    imageReady = false;
    pinnedPoints = clearPoints();
    hoverPoint = null;
    pointerInside = false;
    latestPointer = null;
    lastPromptKey = null;
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
    renderMarkers();
    updatePointControls();
    postWorker({
      type: "clear",
      imageRevision,
      stateRevision,
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
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") setFolderTreeOpen(false);
  });

  function setNavigationMode(mode: NavigationMode): void {
    navigationMode = mode;
    renderDataBrowser();
  }

  positiveTool.addEventListener("click", () => {
    setTool(1);
    if (hoverPoint) {
      lastPromptKey = null;
      submitPrompts(composePrompts(pinnedPoints, hoverPoint));
    }
  });
  negativeTool.addEventListener("click", () => {
    setTool(0);
    if (hoverPoint) {
      lastPromptKey = null;
      submitPrompts(composePrompts(pinnedPoints, hoverPoint));
    }
  });

  undoButton.addEventListener("click", () => {
    pinnedPoints = removeLastPoint(pinnedPoints);
    lastPromptKey = null;
    renderMarkers();
    updatePointControls();
    submitPrompts(composePrompts(pinnedPoints, hoverPoint));
  });

  clearButton.addEventListener("click", () => {
    pinnedPoints = clearPoints();
    hoverPoint = null;
    latestPointer = null;
    lastPromptKey = null;
    renderMarkers();
    updatePointControls();
    submitPrompts([]);
  });
}

function setTool(label: PointLabel): void {
  activeTool = label;
  positiveTool.classList.toggle("active", label === 1);
  positiveTool.setAttribute("aria-pressed", String(label === 1));
  negativeTool.classList.toggle("active", label === 0);
  negativeTool.setAttribute("aria-pressed", String(label === 0));
  toolHint.textContent =
    label === 1
      ? "Positive points include a region."
      : "Negative points exclude a region.";

  if (hoverPoint) {
    hoverPoint = { ...hoverPoint, label };
    lastPromptKey = null;
    updateHoverMarker(hoverPoint);
  }
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
}

function renderMarkers(): void {
  markerLayer.innerHTML = "";
  pinnedPoints.forEach((point) => {
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
  const count = pinnedPoints.length;
  undoButton.disabled = count === 0;
  clearButton.disabled = count === 0;
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
