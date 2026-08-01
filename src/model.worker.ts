/// <reference lib="webworker" />

import type { LayerDescriptor, MainToWorkerMessage, WorkerToMainMessage } from "./protocol";
import { MaskEditor } from "./mask-editor";
import { compositeMasks, excludeOverlaps, outerMaskOutline } from "./mask-compositor";
import { PromptDecodeQueue } from "./prompt-decode-queue";

type LoadImageMessage = Extract<MainToWorkerMessage, { type: "load-image" }>;
type DecodeMessage = Extract<MainToWorkerMessage, { type: "decode" }>;
type SnapshotMessage = Extract<MainToWorkerMessage, { type: "snapshot-annotations" }>;

const worker = self as DedicatedWorkerGlobalScope;
let canvas: OffscreenCanvas | undefined;
let context: OffscreenCanvasRenderingContext2D | null = null;
let pixels: ImageData | undefined;
let serviceReady = false;
let processing = false;
let activeImageId: string | undefined;
let activeImageRevision = -1;
let desiredImageRevision = -1;
let pendingImage: LoadImageMessage | undefined;
const decodeQueue = new PromptDecodeQueue<DecodeMessage>();
let pendingSnapshot: SnapshotMessage | undefined;
let activePrepare: AbortController | undefined;
const latestStateRevision = new Map<string, number>();
const latestCommitRevision = new Map<string, number>();
const maskLayers = new Map<string, WorkerLayer>();
let activeLayerId = "";
let latestMaskLayerId = "";
let preventOverlap = false;
let activeStrokeId = -1;
let activeStrokeLayerId = "";
const HISTORY_BUDGET = 64 * 1024 * 1024;

interface WorkerLayer {
  id: string;
  color: [number, number, number];
  visible: boolean;
  editor: MaskEditor;
  lockedMask?: Uint8Array;
  previewing?: boolean;
}

worker.onmessage = ({ data }: MessageEvent<MainToWorkerMessage>) => {
  switch (data.type) {
    case "initialize":
      if (canvas) return;
      canvas = data.canvas;
      context = canvas.getContext("2d", { alpha: true });
      void watchService();
      return;
    case "load-image":
      desiredImageRevision = data.imageRevision;
      preventOverlap = data.preventOverlap;
      pendingImage = data;
      decodeQueue.clear();
      pendingSnapshot = undefined;
      latestStateRevision.clear();
      activePrepare?.abort();
      clearOverlay();
      void pump();
      return;
    case "decode": {
      const previous = latestStateRevision.get(data.layerId) ?? -1;
      if (data.stateRevision < previous) return;
      latestStateRevision.set(data.layerId, data.stateRevision);
      if (!data.preview) latestCommitRevision.set(data.layerId, data.stateRevision);
      decodeQueue.push(data);
      void pump();
      return;
    }
    case "clear": {
      const previous = latestStateRevision.get(data.layerId) ?? -1;
      if (data.stateRevision < previous) return;
      latestStateRevision.set(data.layerId, data.stateRevision);
      latestCommitRevision.set(data.layerId, data.stateRevision);
      decodeQueue.clearMatching((message) => message.layerId === data.layerId);
      const layer = getLayer(data.layerId);
      latestMaskLayerId = data.layerId;
      layer.editor.clearMask();
      layer.lockedMask = undefined;
      layer.previewing = false;
      renderMasks();
      post({ type: "overlay-cleared", imageRevision: data.imageRevision, layerId: data.layerId, stateRevision: data.stateRevision });
      postEditState(data.imageRevision, data.layerId, 0);
      return;
    }
    case "brush": {
      if (data.imageRevision !== activeImageRevision || data.imageRevision !== desiredImageRevision) return;
      try {
        latestMaskLayerId = data.layerId;
        if (data.phase === "begin") {
          const first = data.points[0];
          if (!first) return;
          commitPreview(getLayer(data.layerId));
          activeStrokeId = data.strokeId;
          activeStrokeLayerId = data.layerId;
          getLayer(data.layerId).editor.beginStroke(data.operation, data.radius, first);
          getLayer(data.layerId).editor.extendStroke(data.points.slice(1));
        } else if (data.strokeId === activeStrokeId && data.layerId === activeStrokeLayerId) {
          if (data.phase === "continue") {
            getLayer(data.layerId).editor.extendStroke(data.points);
          } else {
            getLayer(data.layerId).editor.endStroke(data.points);
            activeStrokeId = -1;
            activeStrokeLayerId = "";
            enforceHistoryBudget();
            postEditState(data.imageRevision, data.layerId, data.editRevision);
          }
        }
        renderMasks();
      } catch (error) {
        postError("edit", error, data.imageRevision);
      }
      return;
    }
    case "invert-mask":
    case "undo-edit":
    case "reset-edits": {
      if (data.imageRevision !== activeImageRevision || data.imageRevision !== desiredImageRevision) return;
      try {
        latestMaskLayerId = data.layerId;
        const editor = getLayer(data.layerId).editor;
        if (data.type === "invert-mask") commitPreview(getLayer(data.layerId));
        if (data.type === "invert-mask") editor.toggleInvert();
        if (data.type === "undo-edit") editor.undo();
        if (data.type === "reset-edits") editor.resetEdits();
        enforceHistoryBudget();
        renderMasks();
        postEditState(data.imageRevision, data.layerId, data.editRevision);
      } catch (error) {
        postError("edit", error, data.imageRevision);
      }
      return;
    }
    case "create-layer": {
      if (data.imageRevision !== activeImageRevision || !canvas) return;
      createWorkerLayer(data.layer, canvas.width, canvas.height);
      renderMasks();
      postEditState(data.imageRevision, data.layer.id, 0);
      return;
    }
    case "update-layer": {
      if (data.imageRevision !== activeImageRevision) return;
      const layer = getLayer(data.layerId);
      if (data.color !== undefined) layer.color = parseColor(data.color);
      if (data.visible !== undefined) {
        layer.visible = data.visible;
        if (!data.visible && layer.id === activeLayerId) restoreLockedMask(layer);
      }
      renderMasks();
      return;
    }
    case "activate-layer": {
      if (data.imageRevision !== activeImageRevision) return;
      decodeQueue.clearPreview();
      for (const layer of maskLayers.values()) restoreLockedMask(layer);
      getLayer(data.layerId);
      activeLayerId = data.layerId;
      activeStrokeId = -1;
      activeStrokeLayerId = "";
      renderMasks();
      postEditState(data.imageRevision, data.layerId, 0);
      return;
    }
    case "delete-layer": {
      if (data.imageRevision !== activeImageRevision) return;
      maskLayers.delete(data.layerId);
      latestStateRevision.delete(data.layerId);
      latestCommitRevision.delete(data.layerId);
      if (activeLayerId === data.layerId) activeLayerId = maskLayers.keys().next().value ?? "";
      if (latestMaskLayerId === data.layerId) latestMaskLayerId = activeLayerId;
      renderMasks();
      return;
    }
    case "set-overlap-prevention": {
      preventOverlap = data.enabled;
      if (data.imageRevision === activeImageRevision && maskLayers.has(data.activeLayerId)) {
        latestMaskLayerId = data.activeLayerId;
        renderMasks();
      }
      return;
    }
    case "cancel-preview": {
      if (data.imageRevision !== activeImageRevision) return;
      const previous = latestStateRevision.get(data.layerId) ?? -1;
      if (data.stateRevision < previous) return;
      latestStateRevision.set(data.layerId, data.stateRevision);
      decodeQueue.clearPreview((message) => message.layerId === data.layerId);
      restoreLockedMask(getLayer(data.layerId));
      renderMasks();
      return;
    }
    case "snapshot-annotations": {
      if (data.imageRevision !== activeImageRevision) return;
      pendingSnapshot = data;
      void pump();
      return;
    }
  }
};

async function watchService(): Promise<void> {
  while (true) {
    try {
      const response = await fetch("/api/status", { cache: "no-store" });
      if (!response.ok) throw new Error(`H100 service returned HTTP ${response.status}.`);
      const status = (await response.json()) as ServiceStatus;
      post({ type: "cache-status", ...status.cache, queueDepth: status.queueDepth,
        currentJob: status.currentJob, backgroundPaused: status.backgroundPaused, activeFolder: status.activeFolder });
      if (status.error) throw new Error(status.error);
      if (status.ready && !serviceReady) {
        serviceReady = true;
        post({ type: "model-ready" });
        void pump();
      } else if (!serviceReady) {
        post({ type: "status", phase: "loading-model", message: "Loading SAM3 on the H100…" });
      }
    } catch (error) {
      if (!serviceReady) {
        postError("initialization", new Error(`${messageOf(error)} Start the backend with “python server/run.py”.`));
        return;
      }
    }
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
}

async function pump(): Promise<void> {
  if (processing || !serviceReady) return;
  processing = true;
  try {
    while (pendingImage || decodeQueue.hasPending() || pendingSnapshot) {
      if (pendingImage) {
        const next = pendingImage;
        pendingImage = undefined;
        decodeQueue.clear();
        await loadImage(next);
      } else if (decodeQueue.hasPending()) {
        const next = decodeQueue.take()!;
        await decode(next);
      } else if (pendingSnapshot) {
        const next = pendingSnapshot;
        pendingSnapshot = undefined;
        snapshotAnnotations(next);
      }
    }
  } finally {
    processing = false;
    if (pendingImage || decodeQueue.hasPending() || pendingSnapshot) void pump();
  }
}

async function loadImage(message: LoadImageMessage): Promise<void> {
  const controller = new AbortController();
  activePrepare = controller;
  post({ type: "status", phase: "encoding-image", message: "Preparing cached H100 embeddings…" });
  try {
    const response = await fetch("/api/images/prepare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ imageId: message.imageId, imageRevision: message.imageRevision }),
      signal: controller.signal,
    });
    if (!response.ok) throw await responseError(response);
    const result = (await response.json()) as PrepareResponse;
    if (message.imageRevision !== desiredImageRevision) return;
    activeImageId = message.imageId;
    activeImageRevision = message.imageRevision;
    prepareOverlay(
      result.width, result.height, message.layers, message.activeLayerId,
      message.restoredMasks, message.latestMaskLayerId,
    );
    post({
      type: "image-ready",
      imageRevision: message.imageRevision,
      width: result.width,
      height: result.height,
      encodeMs: result.prepareMs,
      cacheHit: result.cacheHit,
    });
  } catch (error) {
    if (!(error instanceof DOMException && error.name === "AbortError")) postError("image", error, message.imageRevision);
  } finally {
    if (activePrepare === controller) activePrepare = undefined;
  }
}

async function decode(message: DecodeMessage): Promise<void> {
  if (!activeImageId || message.imageRevision !== activeImageRevision || message.imageRevision !== desiredImageRevision) return;
  const started = performance.now();
  try {
    const response = await fetch("/api/images/segment", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        imageId: activeImageId,
        imageRevision: message.imageRevision,
        stateRevision: message.stateRevision,
        points: message.points,
      }),
    });
    if (!response.ok) throw await responseError(response);
    const imageRevision = headerNumber(response, "X-Image-Revision");
    const stateRevision = headerNumber(response, "X-State-Revision");
    const width = headerNumber(response, "X-Mask-Width");
    const height = headerNumber(response, "X-Mask-Height");
    const serverDecodeMs = headerNumber(response, "X-Decode-Ms");
    const mask = new Uint8Array(await response.arrayBuffer());
    const current =
      imageRevision === activeImageRevision &&
      imageRevision === desiredImageRevision &&
      stateRevision === (message.preview
        ? latestStateRevision.get(message.layerId)
        : latestCommitRevision.get(message.layerId)) &&
      maskLayers.has(message.layerId) &&
      (!message.preview || (activeLayerId === message.layerId && getLayer(message.layerId).visible));
    if (current) drawMask(mask, width, height, message.layerId, message.preview);
    post({
      type: "mask-ready", imageRevision, layerId: message.layerId, stateRevision,
      decodeMs: performance.now() - started, serverDecodeMs, applied: current,
    });
  } catch (error) {
    postError("decode", error, message.imageRevision, message.stateRevision);
  }
}

function prepareOverlay(
  width: number,
  height: number,
  layers: LayerDescriptor[],
  nextActiveLayerId: string,
  restoredMasks: Array<{ layerId: string; mask: Uint8Array }>,
  restoredLatestLayerId?: string,
): void {
  if (!canvas || !context) throw new Error("Overlay canvas is unavailable.");
  canvas.width = width;
  canvas.height = height;
  maskLayers.clear();
  latestStateRevision.clear();
  latestCommitRevision.clear();
  layers.forEach((layer) => createWorkerLayer(layer, width, height));
  if (!maskLayers.has(nextActiveLayerId)) throw new Error("The active mask layer is missing.");
  activeLayerId = nextActiveLayerId;
  latestMaskLayerId = restoredLatestLayerId && maskLayers.has(restoredLatestLayerId)
    ? restoredLatestLayerId : nextActiveLayerId;
  for (const restored of restoredMasks) {
    const layer = getLayer(restored.layerId);
    layer.editor.setBaseMask(restored.mask);
    layer.lockedMask = restored.mask.slice();
  }
  activeStrokeId = -1;
  activeStrokeLayerId = "";
  pixels = context.createImageData(width, height);
  renderMasks();
  for (const layer of maskLayers.values()) postEditState(activeImageRevision, layer.id, 0);
}

function snapshotAnnotations(message: SnapshotMessage): void {
  if (!canvas || message.imageRevision !== activeImageRevision) return;
  for (const layer of maskLayers.values()) restoreLockedMask(layer);
  const raw = new Map<string, Uint8Array>();
  for (const layer of maskLayers.values()) raw.set(layer.id, layer.editor.displayedMask());
  const effective = new Map(raw);
  if (preventOverlap && latestMaskLayerId && effective.has(latestMaskLayerId)) {
    effective.set(
      latestMaskLayerId,
      excludeOverlaps(
        effective.get(latestMaskLayerId)!,
        [...raw.entries()].filter(([id]) => id !== latestMaskLayerId).map(([, mask]) => mask),
        canvas.width * canvas.height,
      ),
    );
  }
  renderMasks();
  post({
    type: "annotation-snapshot",
    requestId: message.requestId,
    imageRevision: message.imageRevision,
    width: canvas.width,
    height: canvas.height,
    latestMaskLayerId,
    layers: [...maskLayers.keys()].map((layerId) => ({
      layerId,
      rawMask: raw.get(layerId)!,
      effectiveMask: effective.get(layerId)!,
    })),
  });
}

function drawMask(mask: Uint8Array, width: number, height: number, layerId: string, preview: boolean): void {
  if (!canvas || !context || !pixels || canvas.width !== width || canvas.height !== height) return;
  const count = width * height;
  if (mask.byteLength !== Math.ceil(count / 8)) throw new Error("The H100 service returned an invalid mask size.");
  const layer = getLayer(layerId);
  latestMaskLayerId = layerId;
  layer.editor.setBaseMask(mask);
  layer.previewing = preview;
  if (!preview) layer.lockedMask = mask.slice();
  renderMasks();
  postEditState(activeImageRevision, layerId, 0);
}

function renderMasks(): void {
  if (!context || !pixels || !canvas) return;
  const displayed = new Map<string, Uint8Array>();
  for (const layer of maskLayers.values()) displayed.set(layer.id, layer.editor.displayedMask());
  const active = maskLayers.get(activeLayerId);
  let committedReference = active?.previewing && active.lockedMask
    ? active.editor.displayedMask(active.lockedMask)
    : undefined;
  if (preventOverlap && latestMaskLayerId && displayed.has(latestMaskLayerId)) {
    const latest = displayed.get(latestMaskLayerId)!;
    const blockers = [...displayed.entries()]
      .filter(([id]) => id !== latestMaskLayerId)
      .map(([, mask]) => mask);
    displayed.set(latestMaskLayerId, excludeOverlaps(latest, blockers, canvas.width * canvas.height));
    if (committedReference && latestMaskLayerId === activeLayerId) {
      committedReference = excludeOverlaps(committedReference, blockers, canvas.width * canvas.height);
    }
  }
  const layers = [];
  for (const layer of maskLayers.values()) {
    if (layer.visible && layer.id !== activeLayerId) {
      layers.push({ mask: displayed.get(layer.id)!, color: layer.color, alpha: 0.34 });
    }
  }
  if (active?.visible) {
    const previewMask = displayed.get(active.id)!;
    const outlinedMask = committedReference ?? previewMask;
    const outlineRadius = Math.max(2, Math.round(Math.min(canvas.width, canvas.height) / 300));
    const edge = outerMaskOutline(outlinedMask, canvas.width, canvas.height, outlineRadius);
    if (committedReference) {
      const committedOnly = excludeOverlaps(
        committedReference,
        [previewMask],
        canvas.width * canvas.height,
      );
      layers.push({ mask: committedOnly, color: active.color, alpha: 0.34 });
      layers.push({ mask: previewMask, color: active.color, alpha: 0.5 });
    } else {
      layers.push({ mask: previewMask, color: active.color, alpha: 0.5 });
    }
    layers.push({ mask: edge, color: [199, 255, 76] as const, alpha: 0.96 });
  }
  compositeMasks(pixels.data, canvas.width * canvas.height, layers);
  context.putImageData(pixels, 0, 0);
}

function clearOverlay(): void {
  maskLayers.clear();
  latestStateRevision.clear();
  latestCommitRevision.clear();
  activeLayerId = "";
  latestMaskLayerId = "";
  activeStrokeId = -1;
  activeStrokeLayerId = "";
  pixels = undefined;
  if (canvas && context) context.clearRect(0, 0, canvas.width, canvas.height);
}

function createWorkerLayer(layer: LayerDescriptor, width: number, height: number): void {
  if (maskLayers.has(layer.id)) throw new Error("A mask layer with this ID already exists.");
  const editor = new MaskEditor(100);
  editor.resetImage(width, height);
  maskLayers.set(layer.id, { id: layer.id, color: parseColor(layer.color), visible: layer.visible, editor });
  latestStateRevision.set(layer.id, 0);
  latestCommitRevision.set(layer.id, 0);
}

function getLayer(layerId: string): WorkerLayer {
  const layer = maskLayers.get(layerId);
  if (!layer) throw new Error("Unknown mask layer.");
  return layer;
}

function restoreLockedMask(layer: WorkerLayer): void {
  if (layer.lockedMask) layer.editor.setBaseMask(layer.lockedMask);
  else layer.editor.clearBaseMask();
  layer.previewing = false;
}

function commitPreview(layer: WorkerLayer): void {
  if (!layer.previewing) return;
  layer.lockedMask = layer.editor.baseMask();
  layer.previewing = false;
}

function parseColor(color: string): [number, number, number] {
  if (!/^#[0-9a-f]{6}$/i.test(color)) throw new Error("Invalid mask layer color.");
  return [
    Number.parseInt(color.slice(1, 3), 16),
    Number.parseInt(color.slice(3, 5), 16),
    Number.parseInt(color.slice(5, 7), 16),
  ];
}

function enforceHistoryBudget(): void {
  let total = 0;
  for (const layer of maskLayers.values()) total += layer.editor.historyBytes();
  while (total > HISTORY_BUDGET) {
    let oldest: WorkerLayer | undefined;
    let oldestSerial = Number.POSITIVE_INFINITY;
    for (const layer of maskLayers.values()) {
      const serial = layer.editor.oldestHistorySerial();
      if (serial !== undefined && serial < oldestSerial) {
        oldest = layer;
        oldestSerial = serial;
      }
    }
    if (!oldest) break;
    const before = oldest.editor.historyBytes();
    oldest.editor.discardOldestHistory();
    total -= before - oldest.editor.historyBytes();
  }
}

function postEditState(imageRevision: number, layerId: string, editRevision: number): void {
  post({ type: "edit-state", imageRevision, layerId, editRevision, ...getLayer(layerId).editor.state() });
}

function headerNumber(response: Response, name: string): number {
  const value = Number(response.headers.get(name));
  if (!Number.isFinite(value)) throw new Error(`Missing response header ${name}.`);
  return value;
}

async function responseError(response: Response): Promise<Error> {
  try {
    const body = (await response.json()) as { detail?: string };
    if (body.detail) return new Error(body.detail);
  } catch { /* Use status fallback. */ }
  return new Error(`H100 service returned HTTP ${response.status}.`);
}

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function post(message: WorkerToMainMessage): void {
  worker.postMessage(message);
}

function postError(
  phase: Extract<WorkerToMainMessage, { type: "error" }>["phase"],
  error: unknown,
  imageRevision?: number,
  stateRevision?: number,
): void {
  post({ type: "error", phase, message: messageOf(error), imageRevision, stateRevision });
}

interface ServiceStatus {
  ready: boolean;
  error: string | null;
  queueDepth: number;
  currentJob: string | null;
  backgroundPaused: boolean;
  activeFolder: { path: string; ready: number; total: number };
  cache: { missing: number; queued: number; encoding: number; ready: number; total: number; gpuResident: number };
}

interface PrepareResponse { width: number; height: number; prepareMs: number; cacheHit: boolean }

export {};
