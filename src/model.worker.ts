/// <reference lib="webworker" />

import type { MainToWorkerMessage, WorkerToMainMessage } from "./protocol";

type LoadImageMessage = Extract<MainToWorkerMessage, { type: "load-image" }>;
type DecodeMessage = Extract<MainToWorkerMessage, { type: "decode" }>;

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
let pendingDecode: DecodeMessage | undefined;
let activePrepare: AbortController | undefined;
const latestStateRevision = new Map<number, number>();

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
      pendingImage = data;
      pendingDecode = undefined;
      latestStateRevision.set(data.imageRevision, 0);
      activePrepare?.abort();
      clearOverlay();
      void pump();
      return;
    case "decode": {
      const previous = latestStateRevision.get(data.imageRevision) ?? -1;
      if (data.stateRevision < previous) return;
      latestStateRevision.set(data.imageRevision, data.stateRevision);
      pendingDecode = data;
      void pump();
      return;
    }
    case "clear": {
      const previous = latestStateRevision.get(data.imageRevision) ?? -1;
      if (data.stateRevision < previous) return;
      latestStateRevision.set(data.imageRevision, data.stateRevision);
      if (pendingDecode?.imageRevision === data.imageRevision) pendingDecode = undefined;
      clearOverlay();
      post({ type: "overlay-cleared", imageRevision: data.imageRevision, stateRevision: data.stateRevision });
    }
  }
};

async function watchService(): Promise<void> {
  while (true) {
    try {
      const response = await fetch("/api/status", { cache: "no-store" });
      if (!response.ok) throw new Error(`H100 service returned HTTP ${response.status}.`);
      const status = (await response.json()) as ServiceStatus;
      post({ type: "cache-status", ...status.cache, queueDepth: status.queueDepth });
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
    while (pendingImage || pendingDecode) {
      if (pendingImage) {
        const next = pendingImage;
        pendingImage = undefined;
        pendingDecode = undefined;
        await loadImage(next);
      } else if (pendingDecode) {
        const next = pendingDecode;
        pendingDecode = undefined;
        await decode(next);
      }
    }
  } finally {
    processing = false;
    if (pendingImage || pendingDecode) void pump();
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
    prepareOverlay(result.width, result.height);
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
    const current = imageRevision === activeImageRevision && imageRevision === desiredImageRevision && stateRevision === latestStateRevision.get(imageRevision);
    if (current) drawMask(mask, width, height);
    post({ type: "mask-ready", imageRevision, stateRevision, decodeMs: performance.now() - started, serverDecodeMs, applied: current });
  } catch (error) {
    postError("decode", error, message.imageRevision, message.stateRevision);
  }
}

function prepareOverlay(width: number, height: number): void {
  if (!canvas || !context) throw new Error("Overlay canvas is unavailable.");
  canvas.width = width;
  canvas.height = height;
  pixels = context.createImageData(width, height);
  for (let index = 0; index < width * height; index += 1) {
    pixels.data[index * 4] = 64;
    pixels.data[index * 4 + 1] = 148;
    pixels.data[index * 4 + 2] = 220;
  }
}

function drawMask(mask: Uint8Array, width: number, height: number): void {
  if (!canvas || !context || !pixels || canvas.width !== width || canvas.height !== height) return;
  const count = width * height;
  if (mask.byteLength !== Math.ceil(count / 8)) throw new Error("The H100 service returned an invalid mask size.");
  for (let index = 0; index < count; index += 1) {
    pixels.data[index * 4 + 3] = (mask[index >> 3]! & (1 << (index & 7))) !== 0 ? 126 : 0;
  }
  context.putImageData(pixels, 0, 0);
}

function clearOverlay(): void {
  if (canvas && context) context.clearRect(0, 0, canvas.width, canvas.height);
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
  cache: { missing: number; queued: number; encoding: number; ready: number; total: number; gpuResident: number };
}

interface PrepareResponse { width: number; height: number; prepareMs: number; cacheHit: boolean }

export {};
