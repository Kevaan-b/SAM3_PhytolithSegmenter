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

export interface LayerDescriptor {
  id: string;
  color: string;
  visible: boolean;
}

export interface RestoredMask {
  layerId: string;
  mask: Uint8Array;
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
      layers: LayerDescriptor[];
      activeLayerId: string;
      preventOverlap: boolean;
      restoredMasks: RestoredMask[];
      latestMaskLayerId?: string;
    }
  | {
      type: "decode";
      imageRevision: number;
      layerId: string;
      stateRevision: number;
      points: PointPrompt[];
      preview: boolean;
    }
  | {
      type: "clear";
      imageRevision: number;
      layerId: string;
      stateRevision: number;
    }
  | {
      type: "brush";
      imageRevision: number;
      layerId: string;
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
      layerId: string;
      editRevision: number;
    }
  | {
      type: "create-layer";
      imageRevision: number;
      layer: LayerDescriptor;
    }
  | {
      type: "update-layer";
      imageRevision: number;
      layerId: string;
      color?: string;
      visible?: boolean;
    }
  | {
      type: "activate-layer" | "delete-layer";
      imageRevision: number;
      layerId: string;
    }
  | {
      type: "cancel-preview";
      imageRevision: number;
      layerId: string;
      stateRevision: number;
    }
  | {
      type: "set-overlap-prevention";
      imageRevision: number;
      enabled: boolean;
      activeLayerId: string;
    }
  | {
      type: "snapshot-annotations";
      requestId: string;
      imageRevision: number;
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
      layerId: string;
      stateRevision: number;
      decodeMs: number;
      serverDecodeMs: number;
      applied: boolean;
    }
  | {
      type: "overlay-cleared";
      imageRevision: number;
      layerId: string;
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
      currentJob: string | null;
      backgroundPaused: boolean;
      activeFolder: { path: string; ready: number; total: number };
    }
  | {
      type: "edit-state";
      imageRevision: number;
      layerId: string;
      editRevision: number;
      hasMask: boolean;
      hasEdits: boolean;
      canUndo: boolean;
      inverted: boolean;
    }
  | {
      type: "annotation-snapshot";
      requestId: string;
      imageRevision: number;
      width: number;
      height: number;
      latestMaskLayerId: string;
      layers: Array<{ layerId: string; rawMask: Uint8Array; effectiveMask: Uint8Array }>;
    };
