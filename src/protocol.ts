export type PointLabel = 0 | 1;

export interface PointPrompt {
  x: number;
  y: number;
  label: PointLabel;
}

export type MainToWorkerMessage =
  | {
      type: "initialize";
      canvas: OffscreenCanvas;
    }
  | {
      type: "load-image";
      imageRevision: number;
      imageId: string;
      url: string;
    }
  | {
      type: "decode";
      imageRevision: number;
      stateRevision: number;
      points: PointPrompt[];
    }
  | {
      type: "clear";
      imageRevision: number;
      stateRevision: number;
    };

export type WorkerToMainMessage =
  | {
      type: "status";
      phase: "loading-model" | "encoding-image";
      message: string;
      progress?: number;
    }
  | {
      type: "model-ready";
    }
  | {
      type: "image-ready";
      imageRevision: number;
      width: number;
      height: number;
      encodeMs: number;
      cacheHit: boolean;
    }
  | {
      type: "mask-ready";
      imageRevision: number;
      stateRevision: number;
      decodeMs: number;
      serverDecodeMs: number;
      applied: boolean;
    }
  | {
      type: "overlay-cleared";
      imageRevision: number;
      stateRevision: number;
    }
  | {
      type: "error";
      phase: "initialization" | "image" | "decode";
      message: string;
      imageRevision?: number;
      stateRevision?: number;
    }
  | {
      type: "cache-status";
      missing: number;
      queued: number;
      encoding: number;
      ready: number;
      total: number;
      gpuResident: number;
      queueDepth: number;
    };
