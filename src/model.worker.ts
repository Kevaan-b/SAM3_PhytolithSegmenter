/// <reference lib="webworker" />

import {
  AutoProcessor,
  env,
  RawImage,
  Sam3TrackerModel,
  Tensor,
} from "@huggingface/transformers";
import { argmax, makePromptData } from "./core";
import type {
  MainToWorkerMessage,
  WorkerToMainMessage,
} from "./protocol";

type LoadImageMessage = Extract<
  MainToWorkerMessage,
  { type: "load-image" }
>;
type DecodeMessage = Extract<MainToWorkerMessage, { type: "decode" }>;
type ProcessorInputs = {
  pixel_values: Tensor;
  original_sizes: [number, number][];
  reshaped_input_sizes: [number, number][];
};
type Embeddings = Record<string, Tensor>;
type ModelOutputs = {
  iou_scores: Tensor;
  pred_masks: Tensor;
  object_score_logits?: Tensor;
};
type ModelLike = Sam3TrackerModel & {
  (inputs: Record<string, Tensor>): Promise<ModelOutputs>;
};
type ProcessorLike = {
  (image: RawImage): Promise<ProcessorInputs>;
  post_process_masks(
    masks: Tensor,
    originalSizes: [number, number][],
    reshapedInputSizes: [number, number][],
  ): Promise<Tensor[]>;
};

const worker = self as DedicatedWorkerGlobalScope;

let overlayCanvas: OffscreenCanvas | undefined;
let overlayContext: OffscreenCanvasRenderingContext2D | null = null;
let overlayPixels: ImageData | undefined;

let model: ModelLike | undefined;
let processor: ProcessorLike | undefined;
let imageInputs: ProcessorInputs | undefined;
let imageEmbeddings: Embeddings | undefined;

let activeImageRevision = -1;
let desiredImageRevision = -1;
let pendingImage: LoadImageMessage | undefined;
let pendingDecode: DecodeMessage | undefined;
let processing = false;
let initialized = false;
const latestStateRevision = new Map<number, number>();

worker.onmessage = ({ data }: MessageEvent<MainToWorkerMessage>) => {
  switch (data.type) {
    case "initialize": {
      if (initialized) return;
      initialized = true;
      overlayCanvas = data.canvas;
      overlayContext = overlayCanvas.getContext("2d", { alpha: true });
      void initializeModel();
      return;
    }

    case "load-image": {
      desiredImageRevision = data.imageRevision;
      pendingImage = data;
      pendingDecode = undefined;
      latestStateRevision.set(data.imageRevision, 0);
      clearOverlay();
      void pump();
      return;
    }

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
      if (
        pendingDecode?.imageRevision === data.imageRevision &&
        pendingDecode.stateRevision <= data.stateRevision
      ) {
        pendingDecode = undefined;
      }
      clearOverlay();
      post({
        type: "overlay-cleared",
        imageRevision: data.imageRevision,
        stateRevision: data.stateRevision,
      });
      return;
    }
  }
};

async function initializeModel(): Promise<void> {
  if (!("gpu" in worker.navigator)) {
    post({
      type: "error",
      phase: "initialization",
      message:
        "WebGPU is unavailable. Use a current Chrome or Edge browser with hardware acceleration enabled.",
    });
    return;
  }

  env.allowRemoteModels = false;
  env.allowLocalModels = true;
  env.localModelPath = "/";

  post({
    type: "status",
    phase: "loading-model",
    message: "Loading the local SAM3 model…",
  });

  try {
    model = (await Sam3TrackerModel.from_pretrained("sam3-q4", {
      device: "webgpu",
      dtype: {
        vision_encoder: "q4",
        prompt_encoder_mask_decoder: "fp32",
      },
      progress_callback: (event) => {
        if (event.status === "progress") {
          post({
            type: "status",
            phase: "loading-model",
            message: `Loading ${shortFilename(event.file)}…`,
            progress: event.progress,
          });
        } else if (event.status === "initiate") {
          post({
            type: "status",
            phase: "loading-model",
            message: `Preparing ${shortFilename(event.file)}…`,
          });
        }
      },
    })) as unknown as ModelLike;
    processor = (await AutoProcessor.from_pretrained(
      "sam3-q4",
    )) as unknown as ProcessorLike;
    post({ type: "model-ready" });
    await pump();
  } catch (error) {
    postError("initialization", error);
  }
}

async function pump(): Promise<void> {
  if (processing || !model || !processor) return;
  processing = true;

  try {
    while (true) {
      if (pendingImage) {
        const nextImage = pendingImage;
        pendingImage = undefined;
        pendingDecode = undefined;
        await loadImage(nextImage);
        continue;
      }

      if (pendingDecode) {
        const nextDecode = pendingDecode;
        pendingDecode = undefined;
        await decode(nextDecode);
        continue;
      }

      break;
    }
  } finally {
    processing = false;
    if (pendingImage || pendingDecode) void pump();
  }
}

async function loadImage(message: LoadImageMessage): Promise<void> {
  if (!model || !processor) return;

  const startedAt = performance.now();
  post({
    type: "status",
    phase: "encoding-image",
    message: "Encoding image once for instant point prompts…",
  });

  try {
    const image = await RawImage.fromURL(
      new URL(message.url, worker.location.origin),
    );
    const nextInputs = (await processor(image)) as ProcessorInputs;
    const nextEmbeddings = await model.get_image_embeddings(nextInputs);
    nextInputs.pixel_values.dispose();

    if (message.imageRevision !== desiredImageRevision) {
      disposeEmbeddings(nextEmbeddings);
      return;
    }

    disposeEmbeddings(imageEmbeddings);
    imageInputs = nextInputs;
    imageEmbeddings = nextEmbeddings;
    activeImageRevision = message.imageRevision;

    const [height, width] = nextInputs.original_sizes[0] ?? [
      image.height,
      image.width,
    ];
    prepareOverlay(width, height);

    post({
      type: "image-ready",
      imageRevision: message.imageRevision,
      width,
      height,
      encodeMs: performance.now() - startedAt,
    });
  } catch (error) {
    postError("image", error, message.imageRevision);
  }
}

async function decode(message: DecodeMessage): Promise<void> {
  if (!model || !processor || !imageInputs || !imageEmbeddings) return;

  const startedAt = performance.now();
  let inputPoints: Tensor | undefined;
  let inputLabels: Tensor | undefined;
  let outputs: ModelOutputs | undefined;
  let masks: Tensor[] | undefined;

  try {
    if (
      message.imageRevision !== activeImageRevision ||
      message.imageRevision !== desiredImageRevision
    ) {
      return;
    }

    if (message.points.length === 0) {
      clearOverlay();
      return;
    }

    const [resizedHeight, resizedWidth] =
      imageInputs.reshaped_input_sizes[0] ?? [1008, 1008];
    const prompt = makePromptData(
      message.points,
      resizedHeight,
      resizedWidth,
    );
    inputPoints = new Tensor(
      "float32",
      prompt.coordinates,
      prompt.pointShape,
    );
    inputLabels = new Tensor("int64", prompt.labels, prompt.labelShape);

    const nextOutputs = await model({
      ...imageEmbeddings,
      input_points: inputPoints,
      input_labels: inputLabels,
    });
    outputs = nextOutputs;

    const nextMasks = await processor.post_process_masks(
      nextOutputs.pred_masks,
      imageInputs.original_sizes,
      imageInputs.reshaped_input_sizes,
    );
    masks = nextMasks;

    const stillCurrent =
      message.imageRevision === activeImageRevision &&
      message.imageRevision === desiredImageRevision &&
      message.stateRevision ===
        latestStateRevision.get(message.imageRevision);

    if (stillCurrent) {
      const selectedCandidate = argmax(
        nextOutputs.iou_scores.data as unknown as ArrayLike<number>,
      );
      drawMask(nextMasks[0], selectedCandidate);
    }

    post({
      type: "mask-ready",
      imageRevision: message.imageRevision,
      stateRevision: message.stateRevision,
      decodeMs: performance.now() - startedAt,
      applied: stillCurrent,
    });
  } catch (error) {
    postError(
      "decode",
      error,
      message.imageRevision,
      message.stateRevision,
    );
  } finally {
    inputPoints?.dispose();
    inputLabels?.dispose();
    outputs?.iou_scores.dispose();
    outputs?.pred_masks.dispose();
    outputs?.object_score_logits?.dispose();
    masks?.forEach((mask) => mask.dispose());
  }
}

function prepareOverlay(width: number, height: number): void {
  if (!overlayCanvas || !overlayContext) {
    throw new Error("The overlay canvas was not initialized.");
  }

  overlayCanvas.width = width;
  overlayCanvas.height = height;
  overlayPixels = overlayContext.createImageData(width, height);

  for (let index = 0; index < width * height; index += 1) {
    const offset = index * 4;
    overlayPixels.data[offset] = 64;
    overlayPixels.data[offset + 1] = 148;
    overlayPixels.data[offset + 2] = 220;
  }
}

function drawMask(mask: Tensor | undefined, candidateIndex: number): void {
  if (!mask || !overlayPixels || !overlayContext || !overlayCanvas) return;

  const pixelCount = overlayCanvas.width * overlayCanvas.height;
  const sourceOffset = candidateIndex * pixelCount;
  const maskData = mask.data;

  if (sourceOffset + pixelCount > maskData.length) {
    throw new Error(
      `Mask tensor shape ${mask.dims.join("×")} does not contain candidate ${candidateIndex}.`,
    );
  }

  for (let index = 0; index < pixelCount; index += 1) {
    overlayPixels.data[index * 4 + 3] = maskData[sourceOffset + index]
      ? 126
      : 0;
  }

  overlayContext.putImageData(overlayPixels, 0, 0);
}

function clearOverlay(): void {
  if (!overlayCanvas || !overlayContext) return;
  overlayContext.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
}

function disposeEmbeddings(embeddings: Embeddings | undefined): void {
  if (!embeddings) return;
  Object.values(embeddings).forEach((embedding) => embedding.dispose());
}

function shortFilename(path: string): string {
  const pieces = path.split("/");
  return pieces.at(-1) ?? path;
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
  post({
    type: "error",
    phase,
    message: error instanceof Error ? error.message : String(error),
    imageRevision,
    stateRevision,
  });
}

export {};
