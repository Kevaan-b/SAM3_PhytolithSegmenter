import "./styles.css";

import {
  clearPoints,
  composePrompts,
  fitDimensions,
  normalizePointer,
  promptKey,
  removeLastPoint,
} from "./core";
import type {
  ExampleImage,
  MainToWorkerMessage,
  PointLabel,
  PointPrompt,
  WorkerToMainMessage,
} from "./protocol";

const EXAMPLES: ExampleImage[] = [
  {
    id: "lumen",
    label: "Example 01",
    description: "Microscopy lumen",
    url: "/example/1/Screenshot%202026-07-22%20at%2012.38.24%E2%80%AFPM.png",
  },
  {
    id: "filament",
    label: "Example 02",
    description: "Microscopy filament",
    url: "/example/2/Screenshot%202026-07-22%20at%2012.32.13%E2%80%AFPM.png",
  },
];

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
        SAM3 Q4 · LOCAL WEBGPU
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
            <h2>Choose an image</h2>
          </div>
          <div class="example-list" id="example-list"></div>
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
            <h2 id="viewer-title">Microscopy lumen</h2>
          </div>
          <div class="status-chip loading" id="status-chip" role="status" aria-live="polite">
            <span class="status-light"></span>
            <span id="status-label">Checking WebGPU…</span>
          </div>
        </div>

        <div class="progress-track" id="progress-track" aria-hidden="true">
          <span id="progress-bar"></span>
        </div>

        <div class="image-frame" id="image-frame">
          <div class="image-stage disabled" id="image-stage">
            <img id="source-image" alt="Selected microscopy example" draggable="false" />
            <canvas id="mask-overlay" aria-hidden="true"></canvas>
            <div class="marker-layer" id="marker-layer" aria-hidden="true"></div>
            <div class="stage-message" id="stage-message">
              <span class="loader"></span>
              <strong>Loading local model</strong>
              <small>The first load includes the 369 MB image encoder.</small>
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

const exampleList = getElement<HTMLDivElement>("example-list");
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

let activeExample = EXAMPLES[0]!;
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
renderExampleButtons();
setExampleImage(activeExample);
updatePointControls();

if (
  !("gpu" in navigator) ||
  !("Worker" in window) ||
  !("transferControlToOffscreen" in overlay)
) {
  showFatal(
    "This demo needs WebGPU and OffscreenCanvas. Open it in a current Chrome or Edge browser with hardware acceleration enabled.",
    "WebGPU unavailable",
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
        loadActiveExample();
        return;
      }

      case "image-ready": {
        if (message.imageRevision !== imageRevision) return;
        setStageSize(message.width, message.height);
        imageReady = true;
        imageStage.classList.remove("disabled");
        stageMessage.classList.add("hidden");
        encodeMetric.textContent = formatDuration(message.encodeMs);
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
        decodeMetric.textContent = formatDuration(message.decodeMs);
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

  function loadActiveExample(): void {
    if (!modelReady) return;
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
      url: activeExample.url,
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

  function chooseExample(example: ExampleImage): void {
    if (example.id === activeExample.id) return;
    activeExample = example;
    imageRevision += 1;
    stateRevision = 0;
    pinnedPoints = clearPoints();
    hoverPoint = null;
    pointerInside = false;
    latestPointer = null;
    lastPromptKey = null;
    encodeMetric.textContent = "—";
    decodeMetric.textContent = "—";
    setExampleImage(example);
    renderExampleButtons();
    renderMarkers();
    updatePointControls();
    loadActiveExample();
  }

  exampleList.addEventListener("click", (event) => {
    const target = (event.target as HTMLElement).closest<HTMLButtonElement>(
      "[data-example-id]",
    );
    const example = EXAMPLES.find(
      ({ id }) => id === target?.dataset.exampleId,
    );
    if (example) chooseExample(example);
  });

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

function renderExampleButtons(): void {
  exampleList.innerHTML = EXAMPLES.map(
    (example, index) => `
      <button
        class="example-button ${example.id === activeExample.id ? "active" : ""}"
        type="button"
        data-example-id="${example.id}"
        aria-pressed="${example.id === activeExample.id}"
      >
        <span class="example-number">0${index + 1}</span>
        <span>
          <strong>${example.label}</strong>
          <small>${example.description}</small>
        </span>
        <span class="example-arrow" aria-hidden="true">↗</span>
      </button>
    `,
  ).join("");
}

function setExampleImage(example: ExampleImage): void {
  sourceImage.src = example.url;
  sourceImage.alt = example.description;
  viewerTitle.textContent = example.description;
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
