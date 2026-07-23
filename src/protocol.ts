export type PointLabel = 0 | 1;

export interface PointPrompt {
  x: number;
  y: number;
  label: PointLabel;
}

export type BrushOperation = "add" | "erase";
export type BrushPhase = "begin" | "continue" | "end";

export interface MaskPoint {
  x: number;
  y: number;
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
    }
  | {
      type: "brush";
      imageRevision: number;
      editRevision: number;
      strokeId: number;
      phase: BrushPhase;
      operation: BrushOperation;
      radius: number;
      points: MaskPoint[];
    }
  | {
      type: "invert-mask" | "undo-edit" | "reset-edits";
      imageRevision: number;
      editRevision: number;
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
      phase: "initialization" | "image" | "decode" | "edit";
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
    }
  | {
      type: "edit-state";
      imageRevision: number;
      editRevision: number;
      hasMask: boolean;
      hasEdits: boolean;
      canUndo: boolean;
      inverted: boolean;
    };
