import type { PointPrompt } from "./protocol";

export const MASK_PALETTE = [
  "#4094dc", "#e06b65", "#34a66f", "#9b72cf",
  "#e29a3b", "#26a6a1", "#d767a7", "#7f8f3f",
] as const;

export interface LayerEditState {
  hasMask: boolean;
  hasEdits: boolean;
  canUndo: boolean;
  inverted: boolean;
}

export interface MaskLayer {
  id: string;
  name: string;
  color: string;
  categoryId: number;
  annotationId?: number;
  visible: boolean;
  pinnedPoints: PointPrompt[];
  stateRevision: number;
  editRevision: number;
  lastPromptKey: string | null;
  editState: LayerEditState;
}

export class MaskLayerCollection {
  private layers: MaskLayer[] = [];
  private activeId = "";
  private nextNumber = 1;

  constructor(private readonly makeId: () => string = defaultId) {
    this.reset();
  }

  reset(categoryId = 1, name = "object", color: string = MASK_PALETTE[0]): MaskLayer {
    this.layers = [];
    this.activeId = "";
    this.nextNumber = 1;
    return this.add(categoryId, name, color);
  }

  add(categoryId = 1, name?: string, color?: string, id?: string, annotationId?: number): MaskLayer {
    const number = this.nextNumber++;
    const layer = createLayer(
      id ?? this.makeId(),
      name ?? `Mask ${number}`,
      color ?? MASK_PALETTE[(number - 1) % MASK_PALETTE.length]!,
      categoryId,
      annotationId,
    );
    this.layers.push(layer);
    this.activeId = layer.id;
    return layer;
  }

  all(): readonly MaskLayer[] { return this.layers; }

  active(): MaskLayer {
    const layer = this.layers.find(({ id }) => id === this.activeId);
    if (!layer) throw new Error("The active mask layer is missing.");
    return layer;
  }

  get(id: string): MaskLayer | undefined { return this.layers.find((layer) => layer.id === id); }

  select(id: string): MaskLayer {
    const layer = this.require(id);
    this.activeId = id;
    return layer;
  }

  delete(id: string): MaskLayer {
    if (this.layers.length === 1) throw new Error("At least one mask layer is required.");
    const index = this.layers.findIndex((layer) => layer.id === id);
    if (index < 0) throw new Error("Unknown mask layer.");
    const wasActive = id === this.activeId;
    this.layers.splice(index, 1);
    if (wasActive) this.activeId = this.layers[Math.min(index, this.layers.length - 1)]!.id;
    return this.active();
  }

  rename(id: string, name: string): void {
    const normalized = name.trim().slice(0, 80);
    if (!normalized) throw new Error("Mask layer names cannot be empty.");
    this.require(id).name = normalized;
  }

  setColor(id: string, color: string): void {
    if (!/^#[0-9a-f]{6}$/i.test(color)) throw new Error("Mask colors must use #RRGGBB format.");
    this.require(id).color = color.toLowerCase();
  }

  setVisible(id: string, visible: boolean): void {
    this.require(id).visible = visible;
  }

  setCategory(id: string, categoryId: number, name: string, color: string): void {
    const layer = this.require(id);
    layer.categoryId = categoryId;
    layer.name = name;
    layer.color = color;
  }

  replace(items: Array<{ id: string; categoryId: number; annotationId?: number; name: string; color: string }>): MaskLayer {
    this.layers = [];
    this.activeId = "";
    this.nextNumber = 1;
    for (const item of items) this.add(item.categoryId, item.name, item.color, item.id, item.annotationId);
    if (this.layers.length === 0) return this.reset();
    this.activeId = this.layers[0]!.id;
    return this.active();
  }

  private require(id: string): MaskLayer {
    const layer = this.layers.find((item) => item.id === id);
    if (!layer) throw new Error("Unknown mask layer.");
    return layer;
  }
}

function createLayer(id: string, name: string, color: string, categoryId: number, annotationId?: number): MaskLayer {
  return {
    id, name, color, categoryId, annotationId, visible: true, pinnedPoints: [], stateRevision: 0, editRevision: 0,
    lastPromptKey: null,
    editState: { hasMask: false, hasEdits: false, canUndo: false, inverted: false },
  };
}

let fallbackId = 0;
function defaultId(): string {
  return globalThis.crypto?.randomUUID?.() ?? `mask-${++fallbackId}`;
}
